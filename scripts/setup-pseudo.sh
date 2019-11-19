./scripts/gdown.pl 'https://drive.google.com/open?id=1jPU9KKIo-QeinhmGPEq6J45Pqiht1rdp' 'data/raw/pseudo.csv'

cd ~/bb/understanding-cloud-organization/data/raw
mkdir joined_images/
cd joined_images
cp -rs ../train_images/* .
cp -rs ../gibs/* .
