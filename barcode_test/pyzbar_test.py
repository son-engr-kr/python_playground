import cv2
from pyzbar import pyzbar

import sys
import cv2
import pylibdmtx.pylibdmtx as dmtx

print("start!")

# img = cv2.imread('barcode_test/qr_test1234.png')
img = cv2.imread('barcode_test/dmtx_test1234.png')
# img = cv2.imread('barcode_test/7.jpg')
# img = cv2.imread('barcode_test/dmtx_crop5.png')


origin_width = int(img.shape[1])
origin_height = int(img.shape[0])
# roi = img[int(origin_height * 0.6):int(origin_height * 0.9), int(origin_width * 0.3):int(origin_width * 0.7)]#y,x
roi = img[:,:]#y,x
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_OTSU)
img_for_barcode = gray
dmtx_resize_factor = 0.6
img_for_dmtx = cv2.resize(gray, (int(origin_width * dmtx_resize_factor), int(origin_height * dmtx_resize_factor)))
# img_for_dmtx = gray[:,:]
result_image = roi

barcodes = pyzbar.decode(img_for_barcode)
for barcode in barcodes:
    (x, y, w, h) = barcode.rect
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    barcodeData = barcode.data.decode("utf-8")
    barcodeType = barcode.type

    
    text = f"barcode: {barcodeType} - {barcodeData}"
    cv2.putText(result_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(text)

data_matrix_decoded = dmtx.decode(img_for_dmtx)
for dmtx_data in data_matrix_decoded:
    (x, y, w, h) = dmtx_data.rect
    y = int(img_for_dmtx.shape[0] - y)
    x = int(x/ dmtx_resize_factor)
    y = int(y/ dmtx_resize_factor)
    w = int(w/ dmtx_resize_factor)
    h = int(h/ dmtx_resize_factor)
    cv2.rectangle(result_image, (x, y), (x + w, y - h), (0, 0, 255), 2)
    
    barcodeData = dmtx_data.data.decode("utf-8")
    barcodeType = "dmtx"

    
    text = f"dmtx: {barcodeType} - {barcodeData}"
    cv2.putText(result_image, text, (x, y -h- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(text)
show_scale = 0.3
width = int(result_image.shape[1] * show_scale)
height = int(result_image.shape[0] * show_scale)
result_resized_img = cv2.resize(result_image, (width, height))
cv2.imshow('result' ,result_resized_img)
cv2.imshow('img_for_barcode' ,img_for_barcode)
cv2.imshow('img_for_dmtx' ,img_for_dmtx)
print(f"data_matrix: {data_matrix_decoded}")
cv2.waitKey(0)