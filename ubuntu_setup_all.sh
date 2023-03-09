# Setup conda env
conda create -n efnet python=3.9 -y && source ~/anaconda3/etc/profile.d/conda.sh && conda activate efnet &&
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y &&
# Install requirements
pip install -r requirements.txt &&
# Install BasicSR package
python setup.py develop --no_cuda_ext

cd datasets/
pip install gdown
# Download Test data
gdown https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view?usp=sharing --fuzzy &&
    unzip GoPro-test-lmdb.zip

# Download Train data
# Crop the train image pairs to 512x512 patches and make the data into lmdb format.
gdown https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing --fuzzy &&
    unzip train.zip && pip install opencv-python tqdm && python scripts/data_preparation/gopro.py
