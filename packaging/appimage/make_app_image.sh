#!/bin/sh
BASE_DIR=$(dirname "$(readlink -f "${0}")")/../../
mkdir -p LiberTEM.AppImage/libertem.AppDir

MC_NAME=Miniconda3-latest-Linux-x86_64.sh
[ ! -f $MC_NAME ] && wget -c -q https://repo.continuum.io/miniconda/$MC_NAME

cd LiberTEM.AppImage/ || exit 1

cd libertem.AppDir || exit 1
HERE=$(dirname $(readlink -f "${0}"))

bash ../../$MC_NAME -b -p ./conda || exit 1
PATH="${HERE}"/conda/bin:$PATH
# conda config --add channels conda-forge
conda create -n libertem python=3.6 -y || exit 1
# FIXME: install specific version (for example from pypi, or continuous build, ...)
pip install "$BASE_DIR"/dist/*.whl || exit 1

rm -rf ./conda/pkgs/

cd .. || exit 1

cp "${BASE_DIR}/corporatedesign/logo/LiberTEM logo icon-512.png" ./libertem-icon-512.png

cat > ./AppRun <<EOF
#!/bin/sh
HERE=$(dirname $(readlink -f "${0}"))
export PATH="${HERE}"/conda/bin:$PATH
libertem-server $*
EOF

chmod a+x ./AppRun

rm ../$MC_NAME

cat > ./LiberTEM.desktop <<EOF
[Desktop Entry]
Type=Application
Terminal=true
Name=LiberTEM
Icon=libertem-icon-512
Exec=LiberTEM %u
Categories=Science;
StartupNotify=true
EOF

echo "AppDir created, creating AppImage..."

cd .. || exit 1

wget -c -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
chmod a+x appimagetool-x86_64.AppImage
./appimagetool-x86_64.AppImage LiberTEM.AppImage

echo "done"
