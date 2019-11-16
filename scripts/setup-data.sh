apt-get install unzip
apt-get install wget

mkdir ~/.kaggle
cd ~/.kaggle
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1miODK84_g8aD7ZBXr3G1uo2ij6cpvawZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/1n/p')&id=1miODK84_g8aD7ZBXr3G1uo2ij6cpvawZ" -O kaggle.json && rm -rf /tmp/cookies.txt

cd ~/bb/understanding-cloud-organization
mkdir -p data/raw
cd data/raw

kaggle competitions download -c understanding_cloud_organization

unzip understanding_cloud_organization.zip
chmod ugo+rwx train.csv
chmod ugo+rwx sample_submission.csv
chmod ugo+rwx train_images.zip
chmod ugo+rwx test_images.zip

mkdir train_images/
unzip train_images.zip -d train_images/
mkdir test_images/
unzip test_images.zip -d test_images/
