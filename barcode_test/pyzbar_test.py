import cv2
from pyzbar import pyzbar

import sys
import cv2
import pylibdmtx.pylibdmtx as dmtx
import time

print("start!")

# img = cv2.imread('barcode_test/test_images/qr_test1234.png')
# img = cv2.imread('barcode_test/test_images/dmtx_test1234.png')
# img = cv2.imread('barcode_test/private_images/7.jpg')
# img = cv2.imread('barcode_test/private_images/dmtx_crop5.png')
img = cv2.imread('barcode_test/test_images/dmtx_rot_test1234.png')


origin_width = int(img.shape[1])
origin_height = int(img.shape[0])
# roi = img[int(origin_height * 0.6):int(origin_height * 0.9), int(origin_width * 0.3):int(origin_width * 0.7)]#y,x
roi = img[:,:]#y,x
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_OTSU)
img_for_zbar = gray
dmtx_resize_factor = 0.6
img_for_dmtx = cv2.resize(gray, (int(origin_width * dmtx_resize_factor), int(origin_height * dmtx_resize_factor)))
# img_for_dmtx = gray[:,:]
result_image = roi
zbar_start = time.time()
barcodes = pyzbar.decode(img_for_zbar)
zbar_end = time.time()
print(f"zbar: {img_for_zbar.shape}, {zbar_end - zbar_start:.5f} sec")
for barcode in barcodes:
    (x, y, w, h) = barcode.rect
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    barcodeData = barcode.data.decode("utf-8")
    barcodeType = barcode.type

    
    text = f"barcode: {barcodeType} - {barcodeData}"
    cv2.putText(result_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(text)
dmtx_start = time.time()
# https://stackoverflow.com/questions/66377973/how-to-improve-pylibdmtx-performance
# data_matrix_decoded = dmtx.decode(img_for_dmtx, max_count=1,threshold=50,min_edge=20,max_edge=60)
data_matrix_decoded = dmtx.decode(img_for_dmtx, max_count=1)#dmtx: (1228, 1468), 0.09405 sec
# data_matrix_decoded = dmtx.decode(img_for_dmtx, max_count=2)#dmtx: (1228, 1468), 10.69253 sec
# data_matrix_decoded = dmtx.decode(img_for_dmtx, max_count=2, threshold=6)#dmtx: (1228, 1468), 12.23962 sec
# data_matrix_decoded = dmtx.decode(img_for_dmtx, max_count=2, shrink=3, threshold=6)#dmtx: (1228, 1468), 1.93698 sec
dmtx_end = time.time()
print(f"dmtx: {img_for_dmtx.shape}, {dmtx_end - dmtx_start:.5f} sec")
for dmtx_data in data_matrix_decoded:
    (x, y, w, h) = dmtx_data.rect
    image_height = img_for_dmtx.shape[0]
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


from pylibdmtx.pylibdmtx import _region, _decoder, _image, _pixel_data, _decoded_matrix_region
from pylibdmtx.wrapper import c_ubyte_p, DmtxPackOrder, DmtxVector2, dmtxMatrix3VMultiplyBy, DmtxUndefined
from ctypes import cast, string_at
from collections import namedtuple
import numpy as np
_pack_order = {
    8: DmtxPackOrder.DmtxPack8bppK,
    16: DmtxPackOrder.DmtxPack16bppRGB,
    24: DmtxPackOrder.DmtxPack24bppRGB,
    32: DmtxPackOrder.DmtxPack32bppRGBX,
}
Decoded = namedtuple('Decoded', 'data rect')
def decode_with_region(image):
    results = []
    pixels, width, height, bpp = _pixel_data(image)
    with _image(cast(pixels, c_ubyte_p), width, height, _pack_order[bpp]) as img:
        with _decoder(img, 1) as decoder:
            while True:
                with _region(decoder, None) as region:
                    if not region:
                        break
                    else:
                        res = _decode_region(decoder, region)
                        if res:
                            open_cv_image = np.array(image)
                            # Convert RGB to BGR
                            # 흑백 이미지인 경우 채널 인덱스를 사용하지 않습니다.
                            if len(open_cv_image.shape) == 3:
                                # Convert RGB to BGR
                                open_cv_image = open_cv_image[:, :, ::-1].copy()
                            
                            height, width = open_cv_image.shape[:2]

                            topLeft = (res.rect['01']['x'], height - res.rect['01']['y'])
                            topRight = (res.rect['11']['x'], height - res.rect['11']['y'])
                            bottomRight = (res.rect['10']['x'], height - res.rect['10']['y'])
                            bottomLeft = (res.rect['00']['x'], height - res.rect['00']['y'])
                            results.append(Decoded(res.data, (topLeft, topRight, bottomRight, bottomLeft)))
    return results

def _decode_region(decoder, region):
    with _decoded_matrix_region(decoder, region, DmtxUndefined) as msg:
        if msg:
            vector00 = DmtxVector2()
            vector11 = DmtxVector2(1.0, 1.0)
            vector10 = DmtxVector2(1.0, 0.0)
            vector01 = DmtxVector2(0.0, 1.0)
            dmtxMatrix3VMultiplyBy(vector00, region.contents.fit2raw)
            dmtxMatrix3VMultiplyBy(vector11, region.contents.fit2raw)
            dmtxMatrix3VMultiplyBy(vector01, region.contents.fit2raw)
            dmtxMatrix3VMultiplyBy(vector10, region.contents.fit2raw)

            return Decoded(
                string_at(msg.contents.output),
                {
                    '00': {
                        'x': int((vector00.X) + 0.5),
                        'y': int((vector00.Y) + 0.5)
                    },
                    '01': {
                        'x': int((vector01.X) + 0.5),
                        'y': int((vector01.Y) + 0.5)
                    },
                    '10': {
                        'x': int((vector10.X) + 0.5),
                        'y': int((vector10.Y) + 0.5)
                    },
                    '11': {
                        'x': int((vector11.X) + 0.5),
                        'y': int((vector11.Y) + 0.5)
                    }
                }
            )
        else:
            return None
        
dmtx_rot_start = time.time()

# 제공된 코드를 사용하여 Data Matrix 코드 해독
data_matrix_rot_decoded = decode_with_region(img_for_dmtx)

dmtx_rot_end = time.time()
print(f"dmtx rot: {img_for_dmtx.shape}, {dmtx_rot_end - dmtx_rot_start:.5f} sec")

# 결과 표시
for dmtx_data in data_matrix_rot_decoded:
    topLeft, topRight, bottomRight, bottomLeft = dmtx_data.rect
    rect_np = np.array(np.array([[topLeft, topRight, bottomRight, bottomLeft]]) / dmtx_resize_factor, dtype=np.int32)
    print(rect_np)
    # 회전된 Data Matrix 코드의 위치를 이미지에 표시
    cv2.polylines(result_image, rect_np, True, (255, 0, 255), 4)

    # 해독된 텍스트 표시
    barcodeData = dmtx_data.data.decode("utf-8")
    text = f"dmtx: {barcodeData}"
    cv2.putText(result_image, text, (int(topLeft[0]/ dmtx_resize_factor), int(topLeft[1]/ dmtx_resize_factor) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(text)





show_scale = 1
width = int(result_image.shape[1] * show_scale)
height = int(result_image.shape[0] * show_scale)
result_resized_img = cv2.resize(result_image, (width, height))
cv2.imshow('result' ,result_resized_img)
cv2.imshow('img_for_barcode' ,img_for_zbar)
cv2.imshow('img_for_dmtx' ,img_for_dmtx)
print(f"data_matrix: {data_matrix_decoded}")
cv2.waitKey(0)