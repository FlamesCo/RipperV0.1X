import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from vq_vae_2 import VQVAE_2 


@st.cache(allow_output_mutation=True)
def load_model():

    device = 'cpu'   #if not torch.cuda.is_available() else 'cuda:0'  # uncomment to run on cpu only

    model = VQVAE_2(in_channel=3, channel=128, n_res_block=2, embedding_dim=64, 
                 decay=0.99, commitment_cost=0.25).to(device)

    checkpoint = torch.load('vqvae-celeba-bw/checkpoint-epoch119-loss1.9094-ppl8.6841.pt', map_location='cpu')  # uncomment to run on cpu only

    model.load_state_dict(checkpoint['model'])

    for param in model.parameters():
        param.requires_grad = False

    return model, device    

    
def load_image(filename):

    img = Image.open(filename)  # .convert('RGB')  # uncomment to convert to RGB if needed (takes longer) 

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])   # resize image and convert to tensor (float32) with values between 0 and 1

    img = transform(img).unsqueeze(0).to('cpu')   # unsqueeze adds a dimension at the beginning of the tensor so that it can be fed into the network (batch size is 1)        

    return img     

    
@st.cache()    
def encode(model, img):      # encode image through trained model and return latent vectors z1 and z2 of shape [1 x embedding dim] each        

    _ , z1, z2, _ , _ = model(img)      # note that we don't need the reconstructions or perceptive loss here        

    return z1[:,:,0], z2[:,:,0]       # we're only interested in the latent vector channels here so select all channels but remove last dimension which is redundant          

    
@st.cache()    					  					  
def decode(model, z1, z2):        	# decode latent vectors into an image through trained model and return reconstructed image r of shape [1 x 3 x 224 x 224]        

      r = model._decode((z1[None],z2[None])).clamp(0., 1.)          ## note that we do need reconstructions here as we are interested in the output image         ## also note that input to _decode must be of type tuple with each element being a tensor of size [batch size x embedding dim]                ## clamp values between 0 and 1 so that it can be displayed with imshow later on        ## for some reason pytorch rounds very small values (<10e-5) to 0 when converting back from tensor to numpy array hence we use .clamp()          ## lastly unsqueeze is used again so that this single image tensor can be fed into imshow later on          r = r[0].permute(1, 2, 0).unsqueeze(0).to('cpu').data          return r              @st                                              def draw():                  try:                      filename = st                            .file_uploader_widget('Upload an image')                      if filename is not None:                          st                            .write('Uploaded `%s`' % filename)                          img = load_image(filename)                          z1, z2 = encode(model, img)                          r = decode(model, z1, z2)                          st                            .image(r, width=224)                  except Exception as e:                      st                            .write(e)                                              if __name__ == '__main__':                      draw()
