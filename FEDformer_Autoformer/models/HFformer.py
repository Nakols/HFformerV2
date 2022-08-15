import torch
import torch.nn as nn
from layers.HFEmbed import DataEmbedding
from layers.HFCorrelation import HFCorrelation, HFCorrelationLayer
from layers.HF_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.seq_len = model_config['seq_len']
        self.label_len = model_config['label_len']
        self.pred_len = model_config['pred_len']
        self.output_attention = model_config['output_attention']
        
        self.lstm_dec1 = nn.LSTM(model_config['d_model'], model_config['d_model'], 4, batch_first=True)
        # self.linear_dec1 = nn.Linear(model_config['d_model'], 3*model_config['d_model'])
        self.activation = nn.PReLU()
        self.linear_dec2 = nn.Linear(model_config['d_model'], model_config['pred_len'])

        # Decomp
        kernel_size = model_config['moving_avg']
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding(model_config['enc_in'], model_config['d_model'], model_config['embed'], model_config['freq'], 
                                           model_config['dropout'])
        self.dec_embedding = DataEmbedding(model_config['dec_in'], model_config['d_model'], model_config['embed'], model_config['freq'],
                                           model_config['dropout'])
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    HFCorrelationLayer(
                        HFCorrelation(False, model_config['factor'], attention_dropout=model_config['dropout'],
                                        output_attention=model_config['output_attention']),
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
            [
                DecoderLayer(
                    HFCorrelationLayer(
                        HFCorrelation(True, model_config['factor'], attention_dropout=model_config['dropout'], output_attention=False),
                        model_config['d_model'], model_config['n_heads']),
                    HFCorrelationLayer(
                        HFCorrelation(False, model_config['factor'], attention_dropout=model_config['dropout'], output_attention=False),
                        model_config['d_model'], model_config['n_heads']),
                    model_config['d_model'],
                    model_config['d_ff'],
                    moving_avg=model_config['moving_avg'],
                    dropout=model_config['dropout'],
                    activation=model_config['activation'],
                )
                for l in range(model_config['d_layers'])
            ],
            norm_layer=my_Layernorm(model_config['d_model']),
            projection=nn.Linear(model_config['d_model'], model_config['c_out'], bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        
        # dec_out = self.linear_dec1(enc_out)
        # print(f'enc_out {enc_out.shape}')
        dec_out, h = self.lstm_dec1(enc_out)
        # print(f'dec_out {dec_out.shape}')
        dec_out = self.activation(dec_out)
        # print(f'dec_out {dec_out.shape}')
        dec_out = self.linear_dec2(dec_out)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
        # dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # print(f'dec out {dec_out.shape}')
        # print(f'enc_out {enc_out.shape}')
        # print(f'trend_init {trend_init.shape}')
        
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        # # final
        # dec_out = trend_part + seasonal_part
        
        # print(f'trend_part {trend_part.shape}')
        # print(f'seasonal_part {seasonal_part.shape}')

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]