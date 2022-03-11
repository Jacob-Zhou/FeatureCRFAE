set -o nounset
set -o errexit
set -o pipefail

# patch fairseq
if [ ! -d elmoformanylangs ];then
    git clone https://github.com/HIT-SCIR/ELMoForManyLangs.git elmoformanylangs_repo
    cd elmoformanylangs_repo
    git checkout -b melmo_for_crfae b3de5f1 
    git apply ../scripts/elmoformanylangs_b3de5f1.patch
    mv elmoformanylangs ../
    cd ..
    rm -rf elmoformanylangs_repo
fi
