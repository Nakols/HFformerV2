import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos,DataEmbedding_wo_pos_temp,DataEmbedding_wo_temp
from layers.AutoCorrelation import  AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.version = model_config['version']
        self.mode_select = model_config['mode_select']
        self.modes = model_config['modes']
        self.seq_len = model_config['seq_len']
        self.label_len = model_config['label_len']
        self.pred_len = model_config['pred_len']
        self.output_attention = model_config['output_attention']

        # Decomp
        kernel_size = model_config['moving_avg']
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if model_config['embed_type'] == 0:
            self.enc_embedding = DataEmbedding_wo_pos(model_config['enc_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                                    model_config['dropout'])
            self.dec_embedding = DataEmbedding_wo_pos(model_config['dec_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                                    model_config['dropout'])
        elif model_config['embed_type'] == 1:
            self.enc_embedding = DataEmbedding(model_config['enc_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                                    model_config['dropout'])
            self.dec_embedding = DataEmbedding(model_config['dec_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                                    model_config['dropout'])
        elif model_config['embed_type'] == 2:
            self.enc_embedding = DataEmbedding_wo_pos_temp(model_config['enc_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                                    model_config['dropout'])
            self.dec_embedding = DataEmbedding_wo_pos_temp(model_config['dec_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                                    model_config['dropout'])
        elif model_config['embed_type'] == 3:
            self.enc_embedding = DataEmbedding_wo_temp(model_config['enc_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                                    model_config['dropout'])
            self.dec_embedding = DataEmbedding_wo_temp(model_config['dec_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                                    model_config['dropout'])

        if model_config['version'] == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=model_config['d_model'], L=model_config['L'], base=model_config['base'])
            decoder_self_att = MultiWaveletTransform(ich=model_config['d_model'], L=model_config['L'], base=model_config['base'])
            decoder_cross_att = MultiWaveletCross(in_channels=model_config['d_model'],
                                                  out_channels=model_config['d_model'],
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=model_config['modes'],
                                                  ich=model_config['d_model'],
                                                  base=model_config['base'],
                                                  activation=model_config['cross_activation'])
        else:
            encoder_self_att = FourierBlock(in_channels=model_config['d_model'],
                                            out_channels=model_config['d_model'],
                                            seq_len=self.seq_len,
                                            modes=model_config['modes'],
                                            mode_select_method=model_config['mode_select'])
            decoder_self_att = FourierBlock(in_channels=model_config['d_model'],
                                            out_channels=model_config['d_model'],
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=model_config['modes'],
                                            mode_select_method=model_config['mode_select'])
            decoder_cross_att = FourierCrossAttention(in_channels=model_config['d_model'],
                                                      out_channels=model_config['d_model'],
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=model_config['modes'],
                                                      mode_select_method=model_config['mode_select'])
        # Encoder
        enc_modes = int(min(model_config['modes'], model_config['seq_len']//2))
        dec_modes = int(min(model_config['modes'], (model_config['seq_len']//2+model_config['pred_len'])//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        model_config['d_model'], model_config['n_heads']),

                    model_config['d_model'],
                    model_config['d_ff'],
                    moving_avg=model_config['moving_avg'],
                    dropout=model_config['dropout'],
                    activation=model_config['activation']
                ) for l in range(model_config['e_layers'])
            ],
            norm_layer=my_Layernorm(model_config['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [ DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        model_config['d_model'], model_config['n_heads']),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        model_config['d_model'], model_config['n_heads']),
                    model_config['d_model'],
                    model_config['c_out'],
                    model_config['d_ff'],
                    moving_avg=model_config['moving_avg'],
                    dropout=model_config['dropout'],
                    activation=model_config['activation'],)
                for l in range(model_config['d_layers']) ],
            
            norm_layer=my_Layernorm(model_config['d_model']),
            projection=nn.Linear(model_config['d_model'], model_config['c_out'], bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
