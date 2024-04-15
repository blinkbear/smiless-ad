mkdir openfaas/
cd openfaas/
git clone https://github.com/openfaas/faas.git
git clone https://github.com/openfaas/faas-netes.git
git clone https://github.com/openfaas/of-watchdog.git
cd faas/
git checkout 4604271076a96ae70b4488afdc516f62fd802dd7
git am  $base_dir/resources/patches/faas.patch
make build-gateway 
docker push blinkbear/openfaas-gateway:latest
cd  $base_dir/openfaas/faas-netes/
git checkout 5065d8b150653dd39d0b693976b446cd9d51b491
git am  $base_dir/resources/patches/faas-netes.patch
make all SERVER=blinkbear IMG_NAME=faas-netes TAG=latest
cd  $base_dir/openfaas/of-watchdog/
git checkout 29909ab030e166461ba1ef663a8a293498c10ebb
git am  $base_dir/resources/patches/of-watchdog.patch
make all SERVER=blinkbear IMG_NAME=of-watchdog TAG=latest
cd $base_dir 
