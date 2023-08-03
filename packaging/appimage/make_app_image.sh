#!/bin/sh
set -ex
BASE_DIR=$(dirname "$(readlink -f "${0}")")/../../

CONDA_PKGS_DIRS=$BASE_DIR/conda-pkgs/


MC_NAME=Miniconda3-latest-Linux-x86_64.sh
[ ! -f $MC_NAME ] && wget -c -q https://repo.continuum.io/miniconda/$MC_NAME

APPDIR=$(dirname "$(readlink -f "${0}")")/AppDir
mkdir -p $APPDIR
cd $APPDIR || exit 1

bash ../$MC_NAME -b -p ./usr || exit 1
PATH="${APPDIR}"/usr/bin:$PATH
# conda config --add channels conda-forge
conda create -n libertem python=3.11 -y || exit 1
# FIXME: install specific version (for example from pypi, or continuous build, ...)

# Build wheel & sdist
( cd "$BASE_DIR" && python -m pip install build && python -m build )

pip install "$BASE_DIR"/dist/*.whl || exit 1

rm -r ./usr/pkgs/
mkdir -p ./usr/share/applications/
mkdir -p ./usr/share/metainfo/
mkdir -p ./usr/share/icons/hicolor/

cp "${BASE_DIR}/packaging/libertem.appdata.xml" ./usr/share/metainfo/
cp -r "${BASE_DIR}/corporatedesign/logo/icons/"* ./usr/share/icons/hicolor/

cp "${BASE_DIR}/corporatedesign/logo/icons/512x512/apps/libertem.png" .

cat > ./AppRun <<\EOF
#!/bin/sh
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
HERE=$(dirname "$(readlink -f "${0}")")
export PATH="${HERE}"/usr/bin:$PATH
python "$HERE/usr/bin/libertem-server" "$@"
EOF

chmod a+x ./AppRun

cat > ./libertem.desktop <<EOF
[Desktop Entry]
Type=Application
Terminal=true
Name=LiberTEM
Comment=Open software platform for pixelated scanning transmission electron microscopy (STEM)
Icon=libertem
Exec=LiberTEM %u
Categories=Science;
StartupNotify=true
EOF

cp ./libertem.desktop ./usr/share/applications/

echo "AppDir created, creating AppImage..."

cd .. || exit 1

wget -c -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
chmod a+x appimagetool-x86_64.AppImage
export VERSION=$(git rev-parse --short HEAD) # linuxdeployqt uses this for naming the file
./appimagetool-x86_64.AppImage "$APPDIR" -g --no-appstream

echo "done"
