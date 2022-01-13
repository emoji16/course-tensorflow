# -*- coding: utf-8 -*-
import tensorflow as tf
from transformer import tf_encode, en_vocab_size, zh_vocab_size, create_padding_mask, scaled_dot_product_attention,\
    create_look_ahead_mask, EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer

'''
demo test
'''

# s3 - demo - test
demo_examples = [
    ("In any case, let’s wish him luck.", "无论如何，让我们祝他好运。"),
    ("The stakes for Africa are enormous.", "这对非洲厉害攸关。")
]
# print(demo_example)

demo_ds = tf.data.Dataset.from_tensor_slices((
    [en for en, _ in demo_examples],[zh for _, zh in demo_examples]
))
demo_ds = demo_ds.map(tf_encode).padded_batch(batch_size=2, padded_shapes=([-1],[-1]))
inp,tar = next(iter(demo_ds))
# print('inp: ',inp) # (2,13)
# print('tar: ',tar) # (2,8)

vocab_size_en = en_vocab_size + 2 + 1
vocab_size_zh = zh_vocab_size + 2 + 1
d_model = 4
embedding_layer_en = tf.keras.layers.Embedding(vocab_size_en, d_model)
embedding_layer_zh = tf.keras.layers.Embedding(vocab_size_zh, d_model)
emb_inp = embedding_layer_en(inp) # (2,13,4)
emb_tar = embedding_layer_zh(tar) # (2,8,4)
# print(emb_inp)
# print(emb_tar)

inp_padding_mask = create_padding_mask(inp) # (2,1,1,13)
mask = tf.squeeze(inp_padding_mask, axis=1) # (2, 1, 13)
q = emb_inp # (2,13,4)
k = emb_inp # (2,13,4)
v = emb_inp # (2,13,4)
_, attention_weights = scaled_dot_product_attention(q,k,v,mask)

seq_len = emb_tar.shape[1] # 2*8*8
look_ahead_mask = create_look_ahead_mask(seq_len) # (2, 8, 8)
# print(look_ahead_mask)
# q = emb_tar # (2,8,4)
# k = emb_tar # (2,8,4)
# v = emb_tar # (2,8,4)

# _, attention_weights = scaled_dot_product_attention(q,k,v,look_ahead_mask)
# print(attention_weights) # 2,13,13 / 2,8,8

num_heads = 2
dff = 8

# mha = MultiHeadAttention(d_model, num_heads)
# output, attention_weights = mha(v,k,q,inp_mask)
# print(output.shape) # 2* 13 * 4
# print(attention_weights.shape) # 2* 2 * 13 * 13

enc_layer = EncoderLayer(d_model, num_heads, dff)
enc_out = enc_layer(emb_inp, training = False, mask = inp_padding_mask)
# print(enc_out.shape) # 2* 13 * 4

tar_padding_mask = create_padding_mask(tar)
look_ahead_mask = create_look_ahead_mask(tar.shape[-1])
combined_mask = tf.maximum(tar_padding_mask, look_ahead_mask)

dec_layer = DecoderLayer(d_model, num_heads, dff)
dec_out,_,_ = dec_layer(emb_tar, enc_out, False, combined_mask, inp_padding_mask)
# print(dec_out.shape) # 2* 8 * 4

# pos_encoding = positional_encoding(tf.constant(50),tf.constant(512))
# plt.pcolormesh(pos_encoding[0],cmap='RdBu')
# plt.xlabel('d_model')
# plt.xlim((0,512))
# plt.xlabel('Position')
# plt.colorbar()
# plt.show()

num_layers = 2
encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size_en)
enc_out = encoder(inp, training = False, mask=inp_padding_mask)
# print(enc_out.shape)
rate = 0.1
decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size_zh,rate)
dec_out = decoder(tar, enc_out, training = False, combined_mask=combined_mask, inp_padding_mask=inp_padding_mask)


tar_inp = tar[:,:-1]
tar_real = tar[:,1:]
tar_padding_mask = create_padding_mask(tar_inp)
look_ahead_mask = create_look_ahead_mask(tar_inp.shape[1])
combined_mask = tf.math.maximum(tar_padding_mask,look_ahead_mask)

transformer = Transformer(num_layers, d_model, num_heads , dff,vocab_size_en,vocab_size_zh)
predictions, _ = transformer(inp, tar_inp, False, inp_padding_mask, combined_mask, inp_padding_mask)
print(tar_real.shape)
print(predictions.shape)