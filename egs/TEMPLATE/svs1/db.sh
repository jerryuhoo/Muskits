# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

KIRITAN=/home/fangzhex/music/data/
NO7SINGING=/home/fangzhex/music/data/
ONIKU=/home/yyu479/svs/data/ONIKU_KURUMI_UTAGOE_DB
OFUTON=/home/yyu479/svs/data/OFUTON_P_UTAGOE_DB
OPENCPOP=/home/yyu479/svs/data/Opencpop
NATSUME=/home/yyu479/svs/data/
NIT_SONG070=/home/yyu479/svs/data/
KISING=/home/yyu479/svs/data/KiSing
NAMINE=/data4/zlt/datas/NAMINE_RITSU_UTAGOE_DB
COMBINE=
CSD=downloads
ITAKO=
POPCS=/home/yyu479/svs/data/popcs

# For only JHU environment
if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
    KIRITAN=/export/c06/jiatong/svs/SVS_system/egs/public_dataset/kiritan/downloads/
    ONIKU=/export/c06/jiatong/svs/data/ONIKU_KURUMI_UTAGOE_DB
    OFUTON=/export/c06/jiatong/svs/data/OFUTON_P_UTAGOE_DB
    NATSUME=/export/c06/jiatong/svs/SVS_system/egs/public_dataset/natsume/downloads/
fi

# For only venus environment
if [[ `hostname` == venus_qt_2241 ]]; then
    KIRITAN=/data3/qt/
    ONIKU=/data3/qt/ONIKU_KURUMI_UTAGOE_DB/
    OFUTON=/data3/qt/OFUTON_P_UTAGOE_DB/
    NATSUME=/data3/qt
    JSUT=/data3/qt
    COMBINE=/data3/qt/Muskits/egs/combine_data/svs1/
fi

if [[ `hostname` == venus_tyx_2235 ]]; then
    KIRITAN=/data3/qt/
    ONIKU=/data3/qt/ONIKU_KURUMI_UTAGOE_DB/
    OFUTON=/data3/qt/OFUTON_P_UTAGOE_DB/
    NATSUME=/data3/qt
    JSUT=/data3/qt
    COMBINE=/data3/qt/Muskits/egs/combine_data/svs1/
    POPCS=/data3/tyx/popcs/
fi

if [[ `hostname` == venus_wyn_2232 ]]; then
    KIRITAN=/data3/qt/
    ONIKU=/data3/qt/ONIKU_KURUMI_UTAGOE_DB/
    OFUTON=/data3/qt/OFUTON_P_UTAGOE_DB/
    NATSUME=/data3/qt
    PJS=/data1/wyn/Mus_data/PJS_corpus_ver1.1/
fi

# For only neptune environment
if [[ `hostname` == neptune_wyn_2245 ]]; then
    PJS=/data1/wyn/Mus_data/PJS_corpus_ver1.1/
    AMEBOSHI=/data1/wyn/Mus_data/ameboshi_ciphyer_utagoe_db/
    OPENCPOP=/data1/wyn/Mus_data/Opencpop/
fi

# For only uranus environment
if [[ `hostname` == uranus_gs_2223 ]]; then
    KIRITAN=/data1/gs/Muskits/egs/kiritan/svs1/data/
    ONIKU=/data1/gs/dataset/ONIKU_KURUMI_UTAGOE_DB
    OFUTON=/data1/gs/dataset/OFUTON_P_UTAGOE_DB
    NATSUME=/data1/gs/dataset/Natsume_Singing_DB
fi

# For only capri environment
if [[ `hostname` == capri_gs_2345 ]]; then
    KIRITAN=/data5/gs/dataset/
    ONIKU=/data5/gs/dataset/ONIKU_KURUMI_UTAGOE_DB
    OFUTON=/data5/gs/dataset/OFUTON_P_UTAGOE_DB
    NATSUME=/data5/gs/dataset/
    COMBINE=/data5/gs/Muskits/egs/combine_data/svs1/
    OPENCPOP=/data5/gs/dataset/Opencpop
fi


if [[ `hostname` == tarus_zlt_2234 ]]; then
    KIRITAN=/data4/zlt/muskit/Muskits-main/egs/kiritan/svs1
    ONIKU=/data4/zlt/muskit/Muskits-main/egs/oniku_kurumi_utagoe_db/svs1/ONIKU_KURUMI_UTAGOE_DB
    OFUTON=/data4/zlt/muskit/Muskits-main/egs/ofuton_p_utagoe_db/svs1/OFUTON_P_UTAGOE_DB
    NAMINE=/data4/zlt/muskit/Muskits-main/egs/namine_ritsu_utagoe_db/svs1/NAMINE_RITSU_UTAGOE_DB
fi
