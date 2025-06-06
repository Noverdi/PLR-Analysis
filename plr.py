#!/usr/bin/env python
import pandas as pd
import numpy as np
import numba as nb
from math import ceil
from typing import Callable
import multiprocessing as mp
import geopandas as gpd
import folium
import branca


def grouper(
        df: pd.DataFrame,  groupby_col: str | list, added_col: list = []
) -> pd.DataFrame:
    """
    Fungsi untuk mengelompokkan data berdasarkan kolom tertentu
    dan mengembalikan agregasi dari kolom-kolom yang ditentukan

    groupby_col :  str | list
        Kolom yang digunakan untuk mengelompokkan data
    df :  pd.DataFrame
        Dataframe yang akan dikelompokkan
    return :  pd.DataFrame
        Dataframe yang sudah dikelompokkan
    """
    col = ['kontribusi', 'tabarru', 'ujroh', 'klaim', 'refund'] + added_col
    df = df.groupby(groupby_col).agg(
        {c: 'sum' for c in col}
    )
    return df


def claim_ratio(
        df: pd.DataFrame,  simple: bool = False,  projected: bool = False
) -> pd.DataFrame:
    """
    Fungsi untuk menghitung rasio klaim

    df :  pd.DataFrame
        Dataframe yang akan dihitung rasio klaimnya
    simple :  bool,  default = False
        jika True,  hanya mengembalikan kolom klaim rasio
        jika False,  mengembalikan dataframe lengkap
    Projected:  bool,  default = False
        jika True,  Rasio klaim = PLR
        Jika False,  Rasio Klaim = klaim / tabarru
    return :  pd.DataFrame
        Dataframe yang sudah ditambahkan kolom rasio klaim
    """
    if not projected:
        df['rasio_klaim'] = (df['klaim'] + df['refund']) / df['tabarru']
        df['rasio_klaim'] = df['rasio_klaim'].fillna(0)
    else:
        df['rasio_klaim'] = (df['klaim'] + df['refund']
                             + df['penyisihan_tabarru']
                             + df['penyisihan_klaim_ibnr']
                             + df['penyisihan_klaim_proses']) / df['tabarru']
        df['rasio_klaim'] = df['rasio_klaim'].fillna(0)

    a = df[['rasio_klaim']].sort_values('rasio_klaim',  ascending=False)
    b = df.sort_values('rasio_klaim',  ascending=False)
    return a if simple else b


def kybmp(df: pd.DataFrame,  simple: bool = False) -> pd.Series | pd.DataFrame:
    """
    Fungsi untuk menghitung KYBMP (Kontribusi Yang Belum Merupakan Pendapatan)

    df :  pd.DataFrame
        Dataframe yang akan dihitung penyisihan KYBMPnya
    simple :  bool
        Jika True,  hanya mengembalikan kolom penyisihan
        Jika False,  mengembalikan dataframe lengkap
    return :  pd.DataFrame
        Series KYBMP atau Dataframe yang ditambahkan kolom penyisihan KYBMP

    """
    df['Penyisihan'] = (df['tgl_akhir'] - df['tgl_val']).dt.days / (
        df['tgl_akhir'] - df['tgl_awal']).dt.days * df['tabarru']

    return df['Penyisihan'] if simple else df


@nb.njit
def os_up(up: float, m: int) -> np.ndarray:
    '''
    Fungsi untuk menghitung sisa pokok uang pertanggungan

    up :  float
        nilai uang pertanggungan
    m :  int
        masa asuransi peserta dalam bulan
    return :  float
        list sisa pokok uang pertanggungan
    '''
    res = np.zeros(m)
    for i in nb.prange(m):
        res[i] = ((m-i) / m) * up
    # res = [( (m-i)/m ) * up for i in range(m)]
    return res


@nb.njit
def penyisihan_tabarru(
        x: int,  m: int,  up: float,  mv: int,  qxList: list,
        iAktu: float = 0.04) -> float:
    '''
        Fungsi untuk menghitung penyisihan tabarru

        x :  int
            usia peserta (dalam tahun)
        m :  int
            masa asuransi peserta (dalam bulan)
        up :  float
            uang pertanggungan peserta
        mv :  int
            bulan valuasi pada saat dilakukan perhitungan penyisihan
        qxList :  list
            list peluang kematian tabel mortalita
        i :  float
            yield investasi yang digunakan untuk mendiskonto proyeksi arus kas
        return :  float
            nilai penyisihan tabarru
    '''

    qx = np.array(qxList)  # list with float value

    lTahun = np.zeros(m, dtype=np.int64)
    lBulan = np.zeros(m, dtype=np.int64)
    for i in nb.prange(m):
        lTahun[i] = ceil((i+1)/12)  # list tahun

    for i in nb.prange(m):
        lBulan[i] = i+1  # list bulan

    lUsia = np.zeros(m, dtype=np.int64)  # list usia
    llx = np.zeros(m)  # list Lx
    ldx = np.zeros(m)  # list dx
    lrx = np.zeros(m)  # list rx

    deathOutgo = np.zeros(m)

    lup = os_up(up, m)  # list sisa up

    liAktu = np.array([iAktu]*30)  # Maksimum Masa asuransi s.d 30 tahun

    discRate = np.zeros(m)
    pvDeathOutgo = np.zeros(m)
    pvDeathOutgo_ret = np.zeros_like(pvDeathOutgo)

    if m < 12:
        pass
    else:
        for i in nb.prange(0,  m):
            lUsia[i] = (x+lTahun[i]-1)
            if i == 0:
                llx[i] = 1
            else:
                llx[i] = llx[i-1]-ldx[i-1]-lrx[i-1]

            ldx[i] = qx[lUsia[i]]/12 * llx[i]
            if i+1 < m:
                lrx[i] = 0
            else:
                lrx[i] = llx[i]-ldx[i]

            deathOutgo[i] = (lup[i]*ldx[i])
            if lBulan[i] <= (mv+12):
                discRate[i] = ((1 + liAktu[0])**(1/12)-1)
            else:
                discRate[i] = (
                    ((1+liAktu[ceil((lBulan[i]-int(mv))/12)-1])**(1/12))-1)

        for i in nb.prange(0,  m):
            if i == 0:
                pvDeathOutgo[i] = ((deathOutgo[m-i-1])/(1+discRate[m-i-1]))
            else:
                pvDeathOutgo[i] = (
                    (deathOutgo[m-i-1]+pvDeathOutgo[i-1])/(1+discRate[m-i-1]))

        for i in nb.prange(len(pvDeathOutgo)):  # reverse list
            pvDeathOutgo_ret[i] = pvDeathOutgo[-(i+1)]

    try:
        if mv < 0:
            return 1
        else:
            res = pvDeathOutgo_ret[int(mv)-1]/llx[int(mv)-1]
            return res
    except Exception:
        return np.nan


def pt_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df['penyisihan_tabarru'] = df.apply(
        lambda x:  penyisihan_tabarru(*x), axis=1)
    return df


def pt_process(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.Series:
    p = mp.Pool(processes=mp.cpu_count())
    # split_dfs = np.array_split(df, mp.cpu_count())
    split_dfs = [
        df.loc[chunk_idx] for chunk_idx in np.array_split(
            df.index,  mp.cpu_count()
        )
    ]
    pool_results = p.map(func,  split_dfs)
    p.close()
    p.join()
    # merging parts processed by different processes
    parts = pd.concat(pool_results,  axis=0)
    return parts['penyisihan_tabarru']


cab_gj = {
    'CAB. ACEH': 'DI. ACEH',
    'CAB. BALIKPAPAN': "KALIMANTAN TIMUR",
    'CAB. BANDUNG': 'JAWA BARAT',
    'CAB. BANJARMASIN': 'KALIMANTAN SELATAN',
    'CAB. BATAM': 'RIAU',
    'CAB. BENGKULU': 'BENGKULU',
    'CAB. BOGOR': 'JAWA BARAT',
    'CAB. CIREBON': 'JAWA BARAT',
    'CAB. DKI JAKARTA': 'DKI JAKARTA',
    'CAB. GORONTALO': 'GORONTALO',
    'CAB. JAMBI': 'JAMBI',
    'CAB. JEMBER': 'JAWA TIMUR',
    'CAB. KEDIRI': 'JAWA TIMUR',
    'CAB. KENDARI': 'SULAWESI TENGGARA',
    'CAB. LAMPUNG': 'LAMPUNG',
    'CAB. LHOKSEUMAWE': 'DI. ACEH',
    'CAB. MAKASSAR': 'SULAWESI SELATAN',
    'CAB. MANADO': 'SULAWESI UTARA',
    'CAB. MATARAM': 'NUSATENGGARA BARAT',
    'CAB. MEDAN': 'SUMATERA UTARA',
    'CAB. MEULABOH': 'DI. ACEH',
    'CAB. PADANG': 'SUMATERA BARAT',
    'CAB. PALANGKARAYA': 'KALIMANTAN TENGAH',
    'CAB. PALEMBANG': 'SUMATERA SELATAN',
    'CAB. PALU': 'SULAWESI TENGAH',
    'CAB. PEKANBARU': 'RIAU',
    'CAB. PONTIANAK': 'KALIMANTAN BARAT',
    'CAB. SAMARINDA': 'KALIMANTAN TIMUR',
    'CAB. SEMARANG': 'JAWA TENGAH',
    'CAB. SURABAYA': 'JAWA TIMUR',
    'CAB. TANGERANG': 'PROBANTEN',
    'CAB. TANGERANG-SERANG': 'PROBANTEN',
    'CAB. TERNATE': 'MALUKU UTARA',
    'CAB. YOGYAKARTA': 'DAERAH ISTIMEWA YOGYAKARTA',
    'KANTOR PUSAT': 'DKI JAKARTA',
    'None': 'None'
}


def get_map(df: pd.DataFrame, json_indo: dict):
    indo = gpd.GeoDataFrame.from_features(
        json_indo,  crs="EPSG: 4326"
    ).iloc[:, [0, 3]]
    df['Provinsi'] = df['cabang'].apply(lambda x: cab_gj[x])

    # menambahkan kolom kontribusi,  tabarru,  klaim,  penyisihan tabarru,
    # penyisihan klaim dan rasio klaim (PLR) ke geodataframe

    col = ['Provinsi']
    prov = grouper(
                df=df.loc[df['Provinsi'] != 'None'],
                groupby_col=col,
                added_col=[
                    'penyisihan_tabarru',
                    'penyisihan_klaim_ibnr',
                    'penyisihan_klaim_proses']
                )
    prov['rasio_klaim'] = claim_ratio(prov, simple=True, projected=True)
    indo_merge = indo.merge(
        prov,
        left_on='Propinsi', right_on='Provinsi', how='left').fillna(0)

    df_loc = pd.DataFrame(
        {"Column":  pd.json_normalize(
            json_indo["features"]
        )["properties.Propinsi"]}
    ).assign(Columnnext=lambda d:  d["Column"].str.len())

    # dictionary rasio klaim
    rk_dict = claim_ratio(
        prov, simple=True, projected=True
        )['rasio_klaim'].to_dict()  # dict rasio klaim
    rk_dict = rk_dict | {v: 0 for v in (
        set(df_loc['Column'].values)-set(rk_dict.keys())
        )}

    colormap = branca.colormap.LinearColormap(  # type:  ignore
        vmin=0,
        vmax=claim_ratio(
            prov, simple=True, projected=True)['rasio_klaim'].max(),
        colors=['green', 'yellow', 'red'],
        caption="Projected Claim Ratio"
        )

    M = folium.Map([-1,  117.5], zoom_start=5)

    popup = folium.GeoJsonPopup(
        fields=[
            "Propinsi",  "kontribusi", 'tabarru',
            'ujroh', 'klaim', 'refund', 'penyisihan_tabarru',
            'penyisihan_klaim_ibnr', 'rasio_klaim'],
        aliases=[
            "Provinsi",  "kontribusi", 'tabarru',
            'ujroh', 'klaim', 'refund',
            'penyisihan_tabarru', 'penyisihan_klaim_ibnr', 'PLR'],
        localize=True,
        labels=True,
        style="background-color:  yellow;",
    )

    tooltip = folium.GeoJsonTooltip(
        fields=[
            "Propinsi",  "kontribusi", 'tabarru',
            'ujroh', 'klaim', 'refund', 'penyisihan_tabarru',
            'penyisihan_klaim_ibnr', 'rasio_klaim'],
        aliases=[
            "Provinsi",  "kontribusi", 'tabarru',
            'ujroh', 'klaim', 'refund', 'penyisihan_tabarru',
            'penyisihan_klaim_ibnr', 'PLR'],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color:  #F0EFEF;
            border:  2px solid black;
            border-radius:  3px;
            box-shadow:  3px;
        """,
        max_width=500,
    )

    folium.GeoJson(
        # json_indo,
        indo_merge,
        zoom_on_click=True,
        style_function=lambda feature:  {
            "fillColor":  colormap(rk_dict[feature["properties"]["Propinsi"]]),
            "color":  "black",
            "weight":  1,
            "dashArray":  "1,  3",
            "fillOpacity":  0.9,
        },
        popup=popup,
        tooltip=tooltip
        ).add_to(M)
    colormap.add_to(M)
    return M


if __name__ == '__main__':
    pass
