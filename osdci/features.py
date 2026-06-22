# osdci/features.py
# Canonical feature column definitions for OSDCI model training.
# Import this in assemble_features.py and train_model.py.

# --- Power infrastructure ---
POWER_FEATURES = [
    "dist_transmission_km",
    "dist_substation_km",
    "operating_capacity_mw_25km",
    "operating_capacity_mw_50km",
    "renewable_capacity_mw_25km",
    "plant_count_25km",
    "dist_nearest_plant_km",
    "colocated_with_power",
]

# --- Grid economics ---
GRID_FEATURES = [
    "elec_price_mwh",
    "is_federal_pma",
    "co2_rate_lb_mwh",        # eGRID subregion
    "renewable_pct",           # eGRID subregion
    "ba_renewable_pct",        # EIA-860 BA level
    "queue_congestion_ratio_ba",
    "operating_capacity_mw_ba",
    "queue_capacity_mw_ba",
    "queue_generator_count_ba",
]

# --- Network connectivity ---
NETWORK_FEATURES = [
    "dist_nearest_ixp_km",
    "ixp_count_500km",
]

# --- Market proximity ---
PROXIMITY_FEATURES = [
    "dist_nearest_city_50k_km",
    "dist_nearest_city_100k_km",
    "dist_nearest_cloud_km",
    "cloud_regions_within_500km",
]

# --- Land cover ---
LAND_FEATURES = [
    "pct_developed",
    "pct_forest",
    "pct_agriculture",
    "pct_shrub_grassland",
    "pct_wetland",
    "pct_barren",
    "mean_impervious_pct",
    "is_built_up",
    "mean_slope_deg",
    "is_flat",
    "is_very_flat",
]

# --- Climate ---
CLIMATE_FEATURES = [
    "mean_temp_c",
    "annual_cdd",
    "cooling_favorable",
]

# --- Water ---
WATER_FEATURES = [
    "water_stress_score",
    "water_stress_cat",
    "high_water_stress",
    "extreme_water_stress",
    "arid_low_use",
    "drought_risk_score",
    "high_drought_risk_raw",
]

# --- Risk ---
RISK_FEATURES = [
    "flood_risk_score",
    "in_floodplain_100yr",
    "in_floodplain_500yr",
    "seismic_risk_score",
    "in_high_seismic",
    "wildfire_risk_score",
    "in_high_wildfire",
]

# --- Population ---
POPULATION_FEATURES = [
    "pop_density_km2",
    "pop_within_25mi",
    "is_rural",
    "is_isolated",
]

# --- Environmental justice ---
EJ_FEATURES = [
    "pm25_pctile",
    "traffic_pctile",
    "cancer_risk_pctile",
    "demographic_index",
    "demographic_index_norm",
    "high_pm25",
    "high_ej_burden",
]

# --- Hard exclusion flags (use as post-prediction masks, not training features) ---
EXCLUSION_FLAGS = [
    "in_protected_area",
    "in_wetland",
    "in_floodplain_100yr",
    "in_tribal_land",
    "in_critical_habitat",
]

# --- All training features (exclusions excluded from model input) ---
ALL_FEATURES = (
    POWER_FEATURES
    + GRID_FEATURES
    + NETWORK_FEATURES
    + PROXIMITY_FEATURES
    + LAND_FEATURES
    + CLIMATE_FEATURES
    + WATER_FEATURES
    + RISK_FEATURES
    + POPULATION_FEATURES
    + EJ_FEATURES
)

# --- Label columns ---
LABEL_COL = "has_dc"
LABEL_META = [
    "facility_count",
    "max_sqft",
    "total_sqft",
    "has_hyperscaler",
    "dominant_type",
    "operators",
    "state_abb",
]

# --- ID columns always kept ---
ID_COLS = ["h3_index", "lat", "lon", "is_conus"]
