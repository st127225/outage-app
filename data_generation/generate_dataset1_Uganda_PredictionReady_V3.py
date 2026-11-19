import os
import random
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TOTAL_SITES = 6000

FIVE_G_SITES = 4900
NON_FIVE_G_SITES = 1100

OUTPUT_DIR = r"D:\Dataset\Uganda_Cell_Telecom_Dataset"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "Dataset_1_Cell_Information_V3.csv"
)

PREVIEW_FILE = os.path.join(
    OUTPUT_DIR,
    "Dataset_1_Cell_Information_V3_Preview.csv"
)


# ============================================================
# 2. UGANDA REGION AND DISTRICT INFORMATION
# ============================================================

REGION_DISTRICTS = {

    "Central": [
        "Kampala",
        "Wakiso",
        "Mukono",
        "Mpigi",
        "Mubende",
        "Luwero"
    ],

    "Eastern": [
        "Jinja",
        "Mbale",
        "Tororo",
        "Iganga",
        "Soroti",
        "Bugiri",
        "Busia",
        "Kamuli"
    ],

    "Northern": [
        "Gulu",
        "Lira",
        "Arua",
        "Kitgum",
        "Yumbe",
        "Moyo",
        "Nebbi"
    ],

    "Western": [
        "Mbarara",
        "Kabale",
        "Kasese",
        "Hoima",
        "Fort Portal",
        "Bushenyi",
        "Masaka"
    ]
}


# ============================================================
# 3. DISTRICT CODES
# ============================================================

DISTRICT_CODES = {

    "Kampala": "KLA",
    "Wakiso": "WAK",
    "Mukono": "MUK",
    "Mpigi": "MPG",
    "Mubende": "MUB",
    "Luwero": "LUW",

    "Jinja": "JIN",
    "Mbale": "MBA",
    "Tororo": "TOR",
    "Iganga": "IGA",
    "Soroti": "SOR",
    "Bugiri": "BUG",
    "Busia": "BUS",
    "Kamuli": "KML",

    "Gulu": "GUL",
    "Lira": "LIR",
    "Arua": "ARU",
    "Kitgum": "KIT",
    "Yumbe": "YUM",
    "Moyo": "MOY",
    "Nebbi": "NEB",

    "Mbarara": "MBR",
    "Kabale": "KAB",
    "Kasese": "KAS",
    "Hoima": "HOI",
    "Fort Portal": "FTP",
    "Bushenyi": "BSH",
    "Masaka": "MSK"
}


# ============================================================
# 4. REALISTIC-LOOKING AREA NAMES
# ============================================================

DISTRICT_AREA_NAMES = {

    "Kampala": [
        "CENTRAL",
        "NAKAWA",
        "MAKINDYE",
        "KAWEMPE",
        "RUBAGA",
        "KISAASI",
        "KOLOLO",
        "NTINDA"
    ],

    "Wakiso": [
        "ENTEBBE",
        "KIRA",
        "KAKIRI",
        "NAMUGONGO",
        "KYALIWAJJALA",
        "NANSANA",
        "KASANGATI"
    ],

    "Mukono": [
        "MUKONO",
        "SEETA",
        "LUGAZI",
        "KATOSI",
        "NAMANOGA"
    ],

    "Mpigi": [
        "MPIGI",
        "BUKAKATA",
        "BULENGA",
        "BUTEKERE"
    ],

    "Mubende": [
        "MUBENDE",
        "KASSANDA",
        "MADDU",
        "KIBOGA"
    ],

    "Luwero": [
        "LUWERO",
        "WOBULENZI",
        "KALULE"
    ],

    "Jinja": [
        "JINJA",
        "BUGEMBE",
        "WANYANGE",
        "KAKIRA"
    ],

    "Mbale": [
        "MBALE",
        "BUGEMA",
        "NAKALOKE",
        "BUSAMAGA"
    ],

    "Tororo": [
        "TORORO",
        "MALABA",
        "MUKUJUU",
        "OSIA"
    ],

    "Iganga": [
        "IGANGA",
        "NAWANDYO",
        "BUGWERI",
        "NAMAAYINGO"
    ],

    "Soroti": [
        "SOROTI",
        "ASWA",
        "ARAPAI",
        "PINGIRE"
    ],

    "Bugiri": [
        "BUGIRI",
        "BUSEDE",
        "BUWUNI"
    ],

    "Busia": [
        "BUSIA",
        "MASAFU",
        "BUTOGOTA"
    ],

    "Kamuli": [
        "KAMULI",
        "BUGULUMBYA",
        "NAMAYINGO"
    ],

    "Gulu": [
        "GULU",
        "LAYIBOR",
        "BUNGATIRA",
        "AWACH"
    ],

    "Lira": [
        "LIRA",
        "ALOI",
        "AMACH",
        "ERUTE"
    ],

    "Arua": [
        "ARUA",
        "VURA",
        "PACHARA"
    ],

    "Kitgum": [
        "KITGUM",
        "MATIDI",
        "LAPONO"
    ],

    "Yumbe": [
        "YUMBE",
        "ARAPI",
        "ODUPI"
    ],

    "Moyo": [
        "MOYO",
        "LAROPI",
        "DUFILE"
    ],

    "Nebbi": [
        "NEBBI",
        "PAKWACH",
        "KOBOKO"
    ],

    "Mbarara": [
        "MBARARA",
        "NYAMITANGA",
        "KAKOBA",
        "KAKYI"
    ],

    "Kabale": [
        "KABALE",
        "RUBANDA",
        "KISORO",
        "RUKIGA"
    ],

    "Kasese": [
        "KASESE",
        "BULEMBIA",
        "KILEMBE",
        "KATWE"
    ],

    "Hoima": [
        "HOIMA",
        "KIGYAMBA",
        "BUSERUKA",
        "BUHIMBA"
    ],

    "Fort Portal": [
        "FORT_PORTAL",
        "KABAROLE",
        "KATUMBA"
    ],

    "Bushenyi": [
        "BUSHENYI",
        "ISHAKA",
        "KASHEKURO",
        "KATOOMA"
    ],

    "Masaka": [
        "MASAKA",
        "NYENDO",
        "BUKAKATA",
        "KIMAANYI"
    ]
}


# ============================================================
# 5. TECHNOLOGY BANDS
# ============================================================

TECH_BANDS = {

    "2G": [
        "GSM900",
        "GSM1800"
    ],

    "3G": [
        "UMTS900",
        "UMTS2100"
    ],

    "4G": [
        "L08",
        "L18",
        "L21",
        "L26"
    ],

    "5G": [
        "N78"
    ]
}


# ============================================================
# 6. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 7. CREATE DISTRICT POOL
# ============================================================

district_pool = []

for region, districts in REGION_DISTRICTS.items():

    for district in districts:

        district_pool.append(
            {
                "Region": region,
                "District": district
            }
        )


# Shuffle the district list
random.shuffle(district_pool)


# Repeat district pool until 6000 sites are available
expanded_district_pool = []

while len(expanded_district_pool) < TOTAL_SITES:

    expanded_district_pool.extend(
        district_pool
    )


expanded_district_pool = (
    expanded_district_pool[:TOTAL_SITES]
)


# Shuffle site locations
random.shuffle(
    expanded_district_pool
)


# ============================================================
# 8. CREATE SITE TYPES
# ============================================================

site_types = (
    ["5G_SITE"] * FIVE_G_SITES
    +
    ["NON_5G_SITE"] * NON_FIVE_G_SITES
)

random.shuffle(site_types)


# ============================================================
# 9. CREATE SITE + CELL RECORDS
# ============================================================

site_cell_records = []

site_to_cells = {}

cell_to_info = {}

site_counter = 1


for site_type, location in zip(
    site_types,
    expanded_district_pool
):

    region = location["Region"]

    district = location["District"]

    district_code = DISTRICT_CODES[
        district
    ]


    # --------------------------------------------------------
    # SITE ID
    # --------------------------------------------------------

    site_id = (
        f"UG-{district_code}-"
        f"{site_counter:04d}"
    )


    # --------------------------------------------------------
    # SITE NAME
    # --------------------------------------------------------

    area_list = DISTRICT_AREA_NAMES[
        district
    ]

    area_name = random.choice(
        area_list
    )

    site_name = (
        f"{district_code}_"
        f"{area_name}_"
        f"{site_counter:04d}"
    )


    # --------------------------------------------------------
    # TECHNOLOGY STRUCTURE
    # --------------------------------------------------------

    if site_type == "5G_SITE":

        # 5G site has exactly 4 cells
        technologies = [
            "2G",
            "3G",
            "4G",
            "5G"
        ]

    else:

        # Non-5G site has exactly 3 cells
        technologies = [
            "2G",
            "3G",
            "4G"
        ]


    cells_for_site = []


    # --------------------------------------------------------
    # CREATE CELLS
    # --------------------------------------------------------

    for cell_number, technology in enumerate(
        technologies,
        start=1
    ):

        band = random.choice(
            TECH_BANDS[technology]
        )


        cell_id = (
            f"{site_id}-"
            f"S{cell_number:02d}"
        )


        cell_name = (
            f"{site_name}_"
            f"S{cell_number:02d}_"
            f"{band}"
        )


        record = {

            "Site_ID": site_id,

            "Site_Name": site_name,

            "Cell_ID": cell_id,

            "Cell_Name": cell_name,

            "Technology": technology,

            "Region": region,

            "District": district,

            "Neighbor_Cell_ID": ""
        }


        site_cell_records.append(
            record
        )


        cells_for_site.append(
            cell_id
        )


        cell_to_info[cell_id] = {

            "Site_ID": site_id,

            "Region": region,

            "District": district,

            "Technology": technology
        }


    site_to_cells[site_id] = (
        cells_for_site
    )


    site_counter += 1


# ============================================================
# 10. CREATE SAME-DISTRICT CELL LOOKUP
# ============================================================

district_cells = {}

for cell_id, info in cell_to_info.items():

    district = info["District"]

    if district not in district_cells:

        district_cells[district] = []

    district_cells[district].append(
        cell_id
    )


# ============================================================
# 11. CREATE NEIGHBOR MAP
#
# RULE:
# - Neighbor must be in SAME DISTRICT
# - Same-site cells can be neighbors
# - Number of neighbors is variable
# - No self-neighbor
# - No duplicates
# ============================================================

neighbor_map = {}


for cell_id, info in cell_to_info.items():

    site_id = info["Site_ID"]

    district = info["District"]


    neighbors = set()


    # --------------------------------------------------------
    # SAME-SITE CELLS
    # --------------------------------------------------------

    same_site_cells = [

        c

        for c in site_to_cells[site_id]

        if c != cell_id
    ]


    # For sites with 4 cells:
    # select 1-2 same-site neighbors
    #
    # For sites with 3 cells:
    # select 1-2 same-site neighbors

    if same_site_cells:

        same_site_count = random.randint(
            1,
            min(2, len(same_site_cells))
        )


        selected_same_site = random.sample(
            same_site_cells,
            same_site_count
        )


        neighbors.update(
            selected_same_site
        )


    # --------------------------------------------------------
    # OTHER CELLS IN SAME DISTRICT
    # --------------------------------------------------------

    same_district_cells = [

        c

        for c in district_cells[district]

        if c != cell_id
        and c not in neighbors
    ]


    # Variable number of district neighbors
    district_count = random.randint(
        2,
        8
    )


    district_candidates = random.sample(

        same_district_cells,

        min(
            district_count,
            len(same_district_cells)
        )
    )


    neighbors.update(
        district_candidates
    )


    # --------------------------------------------------------
    # VARIABLE MAXIMUM
    # --------------------------------------------------------

    maximum_neighbors = random.randint(
        5,
        12
    )


    if len(neighbors) > maximum_neighbors:

        # Keep same-site neighbors first
        same_site_neighbors = [

            n

            for n in neighbors

            if n in same_site_cells
        ]


        other_neighbors = [

            n

            for n in neighbors

            if n not in same_site_cells
        ]


        final_neighbors = set(
            same_site_neighbors
        )


        remaining_slots = (
            maximum_neighbors
            -
            len(final_neighbors)
        )


        if remaining_slots > 0:

            final_neighbors.update(

                random.sample(

                    other_neighbors,

                    min(
                        remaining_slots,
                        len(other_neighbors)
                    )
                )
            )


        neighbors = final_neighbors


    # --------------------------------------------------------
    # GUARANTEE SAME-SITE NEIGHBOR
    # --------------------------------------------------------

    if same_site_cells:

        has_same_site_neighbor = any(

            n in same_site_cells

            for n in neighbors
        )


        if not has_same_site_neighbor:

            # Remove one other neighbor
            # if already at maximum

            if len(neighbors) >= maximum_neighbors:

                removable = [

                    n

                    for n in neighbors

                    if n not in same_site_cells
                ]


                if removable:

                    neighbors.remove(
                        random.choice(
                            removable
                        )
                    )


            neighbors.add(
                random.choice(
                    same_site_cells
                )
            )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    # Remove itself
    neighbors.discard(
        cell_id
    )


    # Keep only valid Cell_IDs
    neighbors = {

        n

        for n in neighbors

        if n in cell_to_info
    }


    neighbor_map[cell_id] = neighbors


# ============================================================
# 12. ADD SOME RECIPROCAL RELATIONSHIPS
#
# Example:
#
# A → B
#
# around 80% chance:
#
# A → B
# B → A
#
# ============================================================

all_cell_ids = list(
    cell_to_info.keys()
)


for cell_id in all_cell_ids:

    current_neighbors = list(
        neighbor_map[cell_id]
    )


    for neighbor in current_neighbors:

        if random.random() < 0.80:

            # IMPORTANT:
            # Neighbor must still be same district

            if (
                cell_to_info[neighbor]["District"]
                ==
                cell_to_info[cell_id]["District"]
            ):

                neighbor_map[
                    neighbor
                ].add(
                    cell_id
                )


# ============================================================
# 13. FINAL NEIGHBOR CLEANING
# ============================================================

for cell_id in all_cell_ids:

    info = cell_to_info[cell_id]

    site_id = info["Site_ID"]

    district = info["District"]


    neighbors = set(
        neighbor_map[cell_id]
    )


    # --------------------------------------------------------
    # Remove self
    # --------------------------------------------------------

    neighbors.discard(
        cell_id
    )


    # --------------------------------------------------------
    # Only same-district neighbors
    # --------------------------------------------------------

    neighbors = {

        n

        for n in neighbors

        if (
            n in cell_to_info
            and
            cell_to_info[n]["District"]
            == district
        )
    }


    # --------------------------------------------------------
    # Same-site cells are calculated for every cell
    # before trimming neighbors (prevents stale loop values).
    # --------------------------------------------------------

    same_site_cells = [
        c for c in site_to_cells[site_id] if c != cell_id
    ]

    # --------------------------------------------------------
    # Maximum 12 neighbors
    # --------------------------------------------------------

    if len(neighbors) > 12:

        same_site_cells = [

            c

            for c in site_to_cells[site_id]

            if c != cell_id
        ]


        same_site_neighbors = [

            n

            for n in neighbors

            if n in same_site_cells
        ]


        other_neighbors = [

            n

            for n in neighbors

            if n not in same_site_cells
        ]


        final_neighbors = set(
            same_site_neighbors
        )


        remaining_slots = (
            12
            -
            len(final_neighbors)
        )


        if remaining_slots > 0:

            final_neighbors.update(

                random.sample(

                    other_neighbors,

                    min(
                        remaining_slots,
                        len(other_neighbors)
                    )
                )
            )


        neighbors = final_neighbors


    # --------------------------------------------------------
    # Ensure same-site neighbor
    # --------------------------------------------------------

    if same_site_cells:

        has_same_site_neighbor = any(

            n in same_site_cells

            for n in neighbors
        )


        if not has_same_site_neighbor:

            if len(neighbors) >= 12:

                removable = [

                    n

                    for n in neighbors

                    if n not in same_site_cells
                ]


                if removable:

                    neighbors.remove(
                        random.choice(
                            removable
                        )
                    )


            neighbors.add(
                random.choice(
                    same_site_cells
                )
            )


    neighbor_map[cell_id] = neighbors


# ============================================================
# 14. PUT NEIGHBOR IDs INTO DATASET
# ============================================================

for record in site_cell_records:

    cell_id = record["Cell_ID"]


    neighbors = list(
        neighbor_map[cell_id]
    )


    # Sort for consistent output
    neighbors.sort()


    record[
        "Neighbor_Cell_ID"
    ] = ",".join(
        neighbors
    )


# ============================================================
# 15. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    site_cell_records
)


# ============================================================
# 16. SHUFFLE ROW ORDER
#
# Timestamps do not exist in Dataset 1,
# so this only changes export order.
# ============================================================

df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(
    drop=True
)


# ============================================================
# 17. SAVE FULL DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. SAVE PREVIEW
# ============================================================

df.head(100).to_csv(
    PREVIEW_FILE,
    index=False
)


# ============================================================
# 19. VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET 1 VALIDATION")
print("=" * 70)


# ============================================================
# BASIC COUNTS
# ============================================================

actual_sites = df[
    "Site_ID"
].nunique()

actual_cells = df[
    "Cell_ID"
].nunique()


expected_cells = (
    FIVE_G_SITES * 4
    +
    NON_FIVE_G_SITES * 3
)


print(
    f"Expected sites : {TOTAL_SITES:,}"
)

print(
    f"Actual sites   : {actual_sites:,}"
)

print(
    f"Expected cells : {expected_cells:,}"
)

print(
    f"Actual cells   : {actual_cells:,}"
)


# ============================================================
# SITE TYPE VALIDATION
# ============================================================

site_technology = (

    df.groupby(
        "Site_ID"
    )["Technology"]

    .apply(list)
)


five_g_site_count = 0

non_five_g_site_count = 0


for technologies in site_technology:

    if "5G" in technologies:

        five_g_site_count += 1

    else:

        non_five_g_site_count += 1


print(
    f"\n5G sites      : {five_g_site_count:,}"
)

print(
    f"Non-5G sites  : {non_five_g_site_count:,}"
)


# ============================================================
# CELLS PER SITE
# ============================================================

cells_per_site = (

    df.groupby(
        "Site_ID"
    )

    .size()
)


print(
    "\nCells per site:"
)

print(
    cells_per_site
    .value_counts()
    .sort_index()
)


# ============================================================
# TECHNOLOGY DISTRIBUTION
# ============================================================

technology_counts = (

    df[
        "Technology"
    ]

    .value_counts()

    .sort_index()
)


print(
    "\nTechnology distribution:"
)

print(
    technology_counts
)


# ============================================================
# REGION DISTRIBUTION
# ============================================================

region_counts = (

    df.groupby(
        "Region"
    )["Site_ID"]

    .nunique()

    .sort_values(
        ascending=False
    )
)


print(
    "\nSites by region:"
)

print(
    region_counts
)


# ============================================================
# DISTRICT DISTRIBUTION
# ============================================================

district_counts = (

    df.groupby(
        "District"
    )["Site_ID"]

    .nunique()

    .sort_values(
        ascending=False
    )
)


print(
    "\nSites by district:"
)

print(
    district_counts
)


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

duplicate_site_ids = (
    df["Site_ID"]
    .duplicated()
    .sum()
)

duplicate_cell_ids = (
    df["Cell_ID"]
    .duplicated()
    .sum()
)

duplicate_site_names = (
    df["Site_Name"]
    .duplicated()
    .sum()
)

duplicate_cell_names = (
    df["Cell_Name"]
    .duplicated()
    .sum()
)


print(
    "\nDuplicate checks:"
)

print(
    "Duplicate Site_ID:",
    duplicate_site_ids
)

print(
    "Duplicate Cell_ID:",
    duplicate_cell_ids
)

print(
    "Duplicate Site_Name:",
    duplicate_site_names
)

print(
    "Duplicate Cell_Name:",
    duplicate_cell_names
)


# ============================================================
# MISSING VALUES
# ============================================================

print(
    "\nMissing values:"
)

print(
    df.isnull().sum()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "Site_ID",
    "Site_Name",
    "Cell_ID",
    "Cell_Name",
    "Technology",
    "Region",
    "District",
    "Neighbor_Cell_ID"
]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns
]


print(
    "\nMissing required columns:",
    missing_columns
)


# ============================================================
# CELL → SITE RELATIONSHIP CHECK
# ============================================================

cell_site_errors = 0


for _, row in df.iterrows():

    expected_prefix = (
        row["Site_ID"] + "-"
    )


    if not row[
        "Cell_ID"
    ].startswith(
        expected_prefix
    ):

        cell_site_errors += 1


print(
    "\nCell/Site relationship errors:",
    cell_site_errors
)


# ============================================================
# TECHNOLOGY COMPOSITION VALIDATION
# ============================================================

composition_errors = 0


for site_id, group in df.groupby(
    "Site_ID"
):

    technologies = set(
        group["Technology"]
    )

    cell_count = len(group)


    # --------------------------------------------------------
    # 5G SITE
    # --------------------------------------------------------

    if "5G" in technologies:

        if cell_count != 4:

            composition_errors += 1


        if technologies != {
            "2G",
            "3G",
            "4G",
            "5G"
        }:

            composition_errors += 1


    # --------------------------------------------------------
    # NON-5G SITE
    # --------------------------------------------------------

    else:

        if cell_count != 3:

            composition_errors += 1


        if technologies != {
            "2G",
            "3G",
            "4G"
        }:

            composition_errors += 1


print(
    "Technology composition errors:",
    composition_errors
)


# ============================================================
# NEIGHBOR VALIDATION
# ============================================================

valid_cell_ids = set(
    df["Cell_ID"]
)


neighbor_counts = []

self_neighbor_errors = 0

duplicate_neighbor_errors = 0

invalid_neighbor_errors = 0

different_district_errors = 0

no_same_site_neighbor_errors = 0


for _, row in df.iterrows():

    cell_id = row[
        "Cell_ID"
    ]

    site_id = row[
        "Site_ID"
    ]

    district = row[
        "District"
    ]


    neighbor_text = row[
        "Neighbor_Cell_ID"
    ]


    if neighbor_text.strip() == "":

        neighbors = []

    else:

        neighbors = [

            n.strip()

            for n
            in neighbor_text.split(",")

            if n.strip()
        ]


    neighbor_counts.append(
        len(neighbors)
    )


    # --------------------------------------------------------
    # SELF NEIGHBOR
    # --------------------------------------------------------

    if cell_id in neighbors:

        self_neighbor_errors += 1


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    if len(neighbors) != len(
        set(neighbors)
    ):

        duplicate_neighbor_errors += 1


    # --------------------------------------------------------
    # VALID CELL IDS
    # --------------------------------------------------------

    for neighbor in neighbors:

        if neighbor not in valid_cell_ids:

            invalid_neighbor_errors += 1


    # --------------------------------------------------------
    # SAME DISTRICT CHECK
    # --------------------------------------------------------

    for neighbor in neighbors:

        if neighbor in cell_to_info:

            neighbor_district = (
                cell_to_info[
                    neighbor
                ]["District"]
            )


            if neighbor_district != district:

                different_district_errors += 1


    # --------------------------------------------------------
    # SAME-SITE NEIGHBOR
    # --------------------------------------------------------

    same_site_cells = [

        c

        for c in site_to_cells[site_id]

        if c != cell_id
    ]


    if same_site_cells:

        has_same_site_neighbor = any(

            n in same_site_cells

            for n in neighbors
        )


        if not has_same_site_neighbor:

            no_same_site_neighbor_errors += 1


# ============================================================
# NEIGHBOR STATISTICS
# ============================================================

print(
    "\nNeighbor statistics:"
)

print(
    f"Minimum neighbors : "
    f"{min(neighbor_counts)}"
)

print(
    f"Maximum neighbors : "
    f"{max(neighbor_counts)}"
)

print(
    f"Average neighbors : "
    f"{sum(neighbor_counts) / len(neighbor_counts):.2f}"
)

print(
    f"Median neighbors  : "
    f"{pd.Series(neighbor_counts).median():.0f}"
)


# ============================================================
# NEIGHBOR ERRORS
# ============================================================

print(
    "\nNeighbor validation:"
)

print(
    "Self-neighbor errors:",
    self_neighbor_errors
)

print(
    "Duplicate neighbor errors:",
    duplicate_neighbor_errors
)

print(
    "Invalid neighbor errors:",
    invalid_neighbor_errors
)

print(
    "Different-district neighbor errors:",
    different_district_errors
)

print(
    "Cells without same-site neighbor:",
    no_same_site_neighbor_errors
)


# ============================================================
# FINAL VALIDATION STATUS
# ============================================================

all_checks_passed = (

    actual_sites == TOTAL_SITES

    and actual_cells == expected_cells

    and five_g_site_count == FIVE_G_SITES

    and non_five_g_site_count == NON_FIVE_G_SITES

    and duplicate_site_ids == 0

    and duplicate_cell_ids == 0

    and duplicate_site_names == 0

    and duplicate_cell_names == 0

    and cell_site_errors == 0

    and composition_errors == 0

    and self_neighbor_errors == 0

    and duplicate_neighbor_errors == 0

    and invalid_neighbor_errors == 0

    and different_district_errors == 0

    and no_same_site_neighbor_errors == 0
)


print(
    "\n" + "=" * 70
)

if all_checks_passed:

    print(
        "ALL VALIDATION CHECKS PASSED ✓"
    )

else:

    print(
        "SOME VALIDATION CHECKS FAILED ✗"
    )

print(
    "=" * 70
)


# ============================================================
# OUTPUT INFORMATION
# ============================================================

print(
    "\nOutput files:"
)

print(
    OUTPUT_FILE
)

print(
    PREVIEW_FILE
)

print(
    "\nDataset 1 generation completed."
)