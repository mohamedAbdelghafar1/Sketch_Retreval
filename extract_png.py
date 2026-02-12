import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

# ==========================
# CONFIGURATION
# ==========================

# change the path to the table and the output image

INPUT_FILE = r'D:\projecs\freelance tasks\Updated model\results3_aligned\model_comparison\comparison_table.csv'# path to the table   
OUTPUT_IMAGE = r'D:\projecs\freelance tasks\Updated model\table_output.png'# path to the output image

FONT_PATH = "arial.ttf"   # Change if needed (e.g., path to your font)
HEADER_FONT_SIZE = 22
CELL_FONT_SIZE = 18

ROW_HEIGHT = 50
CELL_PADDING_X = 15
CELL_PADDING_Y = 10

HEADER_BG_COLOR = (40, 40, 40)
HEADER_TEXT_COLOR = (255, 255, 255)
CELL_BG_COLOR = (255, 255, 255)
ALT_ROW_COLOR = (240, 240, 240)

GRID_COLOR = (180, 180, 180)
TEXT_COLOR = (0, 0, 0)

ALIGNMENT = "center"  # "left", "center", "right"

# ==========================
# LOAD DATA
# ==========================

if INPUT_FILE.endswith(".xlsx"):
    df = pd.read_excel(INPUT_FILE)
else:
    df = pd.read_csv(INPUT_FILE)

columns = df.columns.tolist()
data = df.values.tolist()

# ==========================
# LOAD FONTS
# ==========================

header_font = ImageFont.truetype(FONT_PATH, HEADER_FONT_SIZE)
cell_font = ImageFont.truetype(FONT_PATH, CELL_FONT_SIZE)

# ==========================
# CALCULATE COLUMN WIDTHS
# ==========================

def get_text_size(text, font):
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    return draw.textbbox((0, 0), str(text), font=font)[2:]

column_widths = []

for col_idx, col in enumerate(columns):
    max_width = get_text_size(col, header_font)[0]
    
    for row in data:
        cell_width = get_text_size(row[col_idx], cell_font)[0]
        if cell_width > max_width:
            max_width = cell_width

    column_widths.append(max_width + 2 * CELL_PADDING_X)

# ==========================
# CALCULATE IMAGE SIZE
# ==========================

table_width = sum(column_widths)
table_height = ROW_HEIGHT * (len(data) + 1)

img = Image.new("RGB", (table_width, table_height), "white")
draw = ImageDraw.Draw(img)

# ==========================
# DRAW HEADER
# ==========================

x_offset = 0

for col_idx, col in enumerate(columns):
    width = column_widths[col_idx]

    draw.rectangle(
        [x_offset, 0, x_offset + width, ROW_HEIGHT],
        fill=HEADER_BG_COLOR
    )

    text_width, text_height = get_text_size(col, header_font)

    if ALIGNMENT == "left":
        text_x = x_offset + CELL_PADDING_X
    elif ALIGNMENT == "right":
        text_x = x_offset + width - text_width - CELL_PADDING_X
    else:  # center
        text_x = x_offset + (width - text_width) // 2

    text_y = (ROW_HEIGHT - text_height) // 2

    draw.text((text_x, text_y), str(col),
              font=header_font,
              fill=HEADER_TEXT_COLOR)

    x_offset += width

# ==========================
# DRAW ROWS
# ==========================

for row_idx, row in enumerate(data):
    y_offset = ROW_HEIGHT * (row_idx + 1)

    row_color = ALT_ROW_COLOR if row_idx % 2 == 0 else CELL_BG_COLOR

    x_offset = 0

    for col_idx, cell in enumerate(row):
        width = column_widths[col_idx]

        draw.rectangle(
            [x_offset, y_offset, x_offset + width, y_offset + ROW_HEIGHT],
            fill=row_color
        )

        text_width, text_height = get_text_size(cell, cell_font)

        if ALIGNMENT == "left":
            text_x = x_offset + CELL_PADDING_X
        elif ALIGNMENT == "right":
            text_x = x_offset + width - text_width - CELL_PADDING_X
        else:
            text_x = x_offset + (width - text_width) // 2

        text_y = y_offset + (ROW_HEIGHT - text_height) // 2

        draw.text((text_x, text_y),
                  str(cell),
                  font=cell_font,
                  fill=TEXT_COLOR)

        x_offset += width

# ==========================
# DRAW GRID LINES
# ==========================

# Vertical lines
x = 0
for width in column_widths:
    draw.line([(x, 0), (x, table_height)], fill=GRID_COLOR)
    x += width
draw.line([(table_width - 1, 0), (table_width - 1, table_height)], fill=GRID_COLOR)

# Horizontal lines
for y in range(0, table_height + 1, ROW_HEIGHT):
    draw.line([(0, y), (table_width, y)], fill=GRID_COLOR)

# ==========================
# SAVE IMAGE
# ==========================

img.save(OUTPUT_IMAGE)
print(f"Saved table image to {OUTPUT_IMAGE}")
