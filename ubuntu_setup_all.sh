# Setup conda env
conda create -n efnet python=3.9
conda activate efnet
# Install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Install req
pip install -r requirements.txt
# Install BasicSR package
python setup.py develop --no_cuda_ext

cd datasets/
pip install gdown
# Download Train data
gdown https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing --fuzzy
# Download Test data
gdown https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view?usp=sharing --fuzzy
unzip GoPro-test-lmdb.zip

unzip train.zip
mv train GoPro
# Crop the train image pairs to 512x512 patches and make the data into lmdb format.
python scripts/data_preparation/gopro.py