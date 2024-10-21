import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "total_precipitation",
        "toa_incident_solar_radiation"
    ],
    "year": ["2023"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": ["00:00", "18:00"],
    "data_format": "grib",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()


for monthi in [ "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
	dataset = "reanalysis-era5-pressure-levels"
	request = {
		"product_type": ["reanalysis"],
		"variable": [
			"geopotential",
			"specific_humidity",
			"temperature",
			"u_component_of_wind",
			"v_component_of_wind",
			"vertical_velocity"
		],
		"year": ["2023"],
		"month": [monthi],
		"day": [
			"01", "02", "03",
			"04", "05", "06",
			"07", "08", "09",
			"10", "11", "12",
			"13", "14", "15",
			"16", "17", "18",
			"19", "20", "21",
			"22", "23", "24",
			"25", "26", "27",
			"28", "29", "30",
			"31"
		],
		"time": ["00:00", "18:00"],
		"pressure_level": [
			"50", "100", "150",
			"200", "250", "300",
			"400", "500", "600",
			"700", "850", "925",
			"1000"
		],
		"data_format": "grib",
		"download_format": "unarchived"
	}
	
	client = cdsapi.Client()
	client.retrieve(dataset, request).download()


