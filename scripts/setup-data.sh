apt-get install unzip
apt-get install wget

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
