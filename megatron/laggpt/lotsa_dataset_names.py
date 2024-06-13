BUILDINGS_BENCH = [
    "bdg-2_panther",
    "bdg-2_fox",
    "bdg-2_rat",
    "bdg-2_bear",
    "lcl",
    "smart",
    "ideal",
    "sceaux",
    "borealis",
    "buildings_900k",
]

CLIMATE_LEARN = [
    # "cmip6",
    "cmip6_1850",
    "cmip6_1855",
    "cmip6_1860",
    "cmip6_1865",
    "cmip6_1870",
    "cmip6_1875",
    "cmip6_1880",
    "cmip6_1885",
    "cmip6_1890",
    "cmip6_1895",
    "cmip6_1900",
    "cmip6_1905",
    "cmip6_1910",
    "cmip6_1915",
    "cmip6_1920",
    "cmip6_1925",
    "cmip6_1930",
    "cmip6_1935",
    "cmip6_1940",
    "cmip6_1945",
    "cmip6_1950",
    "cmip6_1955",
    "cmip6_1960",
    "cmip6_1965",
    "cmip6_1970",
    "cmip6_1975",
    "cmip6_1980",
    "cmip6_1985",
    "cmip6_1990",
    "cmip6_1995",
    "cmip6_2000",
    "cmip6_2005",
    "cmip6_2010",
    # "era5",
    "era5_1989",
    "era5_1990",
    "era5_1991",
    "era5_1992",
    "era5_1993",
    "era5_1994",
    "era5_1995",
    "era5_1996",
    "era5_1997",
    "era5_1998",
    "era5_1999",
    "era5_2000",
    "era5_2001",
    "era5_2002",
    "era5_2003",
    "era5_2004",
    # "era5_2005", # Doesn"t seem to exist?
    "era5_2006",
    "era5_2007",
    "era5_2008",
    "era5_2009",
    "era5_2010",
    "era5_2011",
    "era5_2012",
    "era5_2013",
    "era5_2014",
    "era5_2015",
    "era5_2016",
    "era5_2017",
    "era5_2018",
]

CLOUDOPS = [
    "azure_vm_traces_2017",
    "borg_cluster_data_2011",
    "alibaba_cluster_trace_2018",
]

GLUONTS = [
    "taxi_30min",
    "uber_tlc_daily",
    "uber_tlc_hourly",
    "wiki-rolling_nips",
    "m5",
]

LARGE_ST = [
    # "largest",
    "largest_2017",
    "largest_2018",
    "largest_2019",
    "largest_2020",
    "largest_2021",
]

LIB_CITY = [
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
    "PEMS_BAY",
    "LOS_LOOP",
    "LOOP_SEATTLE",
    "SZ_TAXI",
    "BEIJING_SUBWAY_30MIN",
    "SHMETRO",
    "HZMETRO",
    "Q-TRAFFIC",
]

MONASH = [
    "london_smart_meters_with_missing",
    "wind_farms_with_missing",
    "wind_power",
    "solar_power",
    "oikolab_weather",
    "elecdemand",
    "covid_mobility",
    "kaggle_web_traffic_weekly",
    "extended_web_traffic_with_missing",
    "m1_yearly",
    "m1_quarterly",
    "m1_monthly",
    "monash_m3_yearly",
    "monash_m3_quarterly",
    "monash_m3_monthly",
    "monash_m3_other",
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "nn5_daily_with_missing",
    "nn5_weekly",
    "tourism_yearly",
    "tourism_quarterly",
    "tourism_monthly",
    "cif_2016_12",
    "cif_2016_6",
    "traffic_weekly",
    "traffic_hourly",
    "australian_electricity_demand",
    "rideshare_with_missing",
    "saugeenday",
    "sunspot_with_missing",
    "temperature_rain_with_missing",
    "vehicle_trips_with_missing",
    "weather",
    "car_parts_with_missing",
    "fred_md",
    "pedestrian_counts",
    "hospital",
    "covid_deaths",
    "kdd_cup_2018_with_missing",
    "bitcoin_with_missing",
    "us_births",
]

# Monash datasets excluding:
# london_smart_meters_with_missing, wind_farms_with_missing, wind_power, solar_power, oikolab_weather, covid_mobility - REASON: Not originally in Monash benchmark
# m1_yearly, m1_quarterly, monash_m3_yearly, monash_m3_quarterly, m4_yearly, m4_quarterly, tourism_yearly - REASON: Too short to split
# extended_web_traffic_with_missing, kaggle_web_traffic_weekly - REASON: Contains overlap with original Web Traffic dataset
IN_DISTR_EVAL = [
    "elecdemand",
    "m1_monthly",
    "monash_m3_monthly",
    "monash_m3_other",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "nn5_daily_with_missing",
    "nn5_weekly",
    "tourism_quarterly",
    "tourism_monthly",
    "cif_2016_12",
    "cif_2016_6",
    "traffic_weekly",
    "traffic_hourly",
    "australian_electricity_demand",
    "rideshare_with_missing",
    "saugeenday",
    "sunspot_with_missing",
    "temperature_rain_with_missing",
    "vehicle_trips_with_missing",
    "weather",
    "car_parts_with_missing",
    "fred_md",
    "pedestrian_counts",
    "hospital",
    "covid_deaths",
    "kdd_cup_2018_with_missing",
    "bitcoin_with_missing",
    "us_births",
]

PRO_EN_FO = [
    "covid19_energy",
    "gfc12_load",
    "gfc14_load",
    "gfc17_load",
    "pdb",
    "spain",
    "hog",
    "bull",
    "cockatoo",
    "elf",
]

SUBSEASONAL_CLIMATE_USA = [
    "subseasonal",
    "subseasonal_precip",
]

MISC = [
    "kdd2022",
    "godaddy",
    "favorita_sales",
    "favorita_transactions",
    "restaurant", 
    "hierarchical_sales",
    "china_air_quality",
    "beijing_air_quality",
    "residential_load_power",
    "residential_pv_power",
    "cdc_fluview_ilinet",
    "cdc_fluview_who_nrevss",
    "project_tycho",
    "M_DENSE", # Rotterdam? Not sure what this dataset is
]

DEBUG = [
    "BEIJING_SUBWAY_30MIN",
    "HZMETRO",
    "SZ_TAXI",
]

DEFAULT = (
    BUILDINGS_BENCH + \
    CLIMATE_LEARN + \
    CLOUDOPS + \
    GLUONTS + \
    LARGE_ST + \
    LIB_CITY + \
    MONASH + \
    PRO_EN_FO + \
    SUBSEASONAL_CLIMATE_USA + \
    MISC
)
        