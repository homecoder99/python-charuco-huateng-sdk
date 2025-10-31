# -*- coding: utf-8 -*-
"""
Huateng/MindVision MVSDK + OpenCV(aruco) 라이브 캘리브레이션 스크립트
- 2대 카메라(좌/우)를 MVSDK Python으로 동시에 소프트 트리거 캡처
- ChArUco 보드로 단안(좌/우) 내·외부 파라미터 보정 + 스테레오 보정/정렬
- 리맵 맵(rectify_maps_*.npz) 및 결과(stereo.yml, intrinsics_*.yml) 저장

필수:
  - MVSDK Python 바인딩(mvsdk.py)과 libMVSDK.so가 import/로드 가능해야 함
  - OpenCV "contrib" 버전(aruco 모듈 포함): opencv-contrib-python>=4.7 권장

설치 예시(가상환경 권장):
  python3 -m venv ~/venvs/cv && source ~/venvs/cv/bin/activate
  pip install --upgrade pip
  pip install opencv-contrib-python numpy
  # SDK 경로 설정(예시)
  export PYTHONPATH=$PYTHONPATH:~/MVSDK/python
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/MVSDK/lib/x64

실행 예시:
  python stereo_charuco_mvsdk_live.py \
    --pairs 20 \
    --dict 5X5_1000 --squaresX 10 --squaresY 12 --square 20.0 --marker 14.0 \
    --min-corners 20 --exposure-us 30000 --out out_live --debug

사용법:
  - 미리 2대 카메라 전원/네트워크 연결 후 스크립트 실행
  - 미预览 창(LEFT/RIGHT)이 뜨면 보드를 다양한 위치/각도로 비춤
  - 키보드 [SPACE] 또는 [c]: 한 쌍 캡처(양쪽에서 ChArUco 충분히 보일 때만 채택)
  - [s]: 현재 프레임을 강제로 저장(감지 부족해도 원본만 저장)
  - [x]: 좌/우 카메라 스왑
  - [q]: 종료(필요 쌍 수를 모으면 자동으로 보정 수행 후 저장)
"""
import os
import sys
import time
import json
import glob
import ctypes
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
# Qt Wayland 플러그인 이슈 회피: X11(xcb) 백엔드로 강제
os.environ.setdefault('QT_QPA_PLATFORM','xcb')
import cv2

import mvsdk  # Huateng/MindVision Python SDK

# --- CameraInit 재시도 유틸: -37(네트워크 전송오류), -13(서브넷/점유) 등 대응 ---
def _init_with_retry(dev, retries=4, base_sleep=0.2):
    last = None
    for k in range(retries):
        try:
            return mvsdk.CameraInit(dev, -1, -1)
        except mvsdk.CameraException as e:
            s = f"{e.error_code} {getattr(e, 'message', '')}"
            # -37: 网络数据发送错误, -13: 이미 점유/서브넷, 기타 네트워크성 실패 재시도
            if (getattr(e, 'error_code', None) in (-37, -13)) or ("网络" in s) or ("send" in s.lower()):
                time.sleep(base_sleep * (k + 1))
                continue
            last = e
            break
        except Exception as e:
            last = e
            break
    if last:
        raise last
    raise RuntimeError("Unknown init failure")

# ---------------------- ArUco/ChArUco 호환 헬퍼 ----------------------
def get_aruco_dict(name):
    ar = cv2.aruco
    if isinstance(name, str):
        name = name.upper()
        table = {
            "4X4_50": ar.DICT_4X4_50,
            "4X4_100": ar.DICT_4X4_100,
            "5X5_1000": ar.DICT_5X5_1000,
            "6X6_1000": ar.DICT_6X6_1000,
            "7X7_1000": ar.DICT_7X7_1000,
            "APRILTAG_36H11": getattr(ar, "DICT_APRILTAG_36h11", getattr(ar, "DICT_APRILTAG_36H11", None)),
        }
        assert name in table and table[name] is not None, f"Unknown dictionary: {name}"
        dict_id = table[name]
    else:
        dict_id = name
    if hasattr(ar, "getPredefinedDictionary"):
        return ar.getPredefinedDictionary(dict_id)
    return ar.Dictionary_get(dict_id)


def create_charuco_board(squaresX, squaresY, square_mm, marker_mm, aruco_dict):
    ar = cv2.aruco
    # --- OpenCV 버전별 생성자 호환 ---
    # 순서: (A) 파이썬 생성자 호출 → (B) 정적 create() → (C) _create() 팩토리
    # A) CharucoBoard((cols, rows), square, marker, dict)
    if hasattr(ar, "CharucoBoard"):
        # 일부 4.x 빌드에서는 파이썬 생성자 호출이 가능
        try:
            return ar.CharucoBoard((int(squaresX), int(squaresY)), float(square_mm), float(marker_mm), aruco_dict)
        except Exception:
            pass
        # B) CharucoBoard.create(...)
        try:
            if hasattr(ar.CharucoBoard, "create"):
                return ar.CharucoBoard.create(int(squaresX), int(squaresY), float(square_mm), float(marker_mm), aruco_dict)
        except Exception:
            pass
    # C) CharucoBoard_create(...)
    if hasattr(ar, "CharucoBoard_create"):
        return ar.CharucoBoard_create(int(squaresX), int(squaresY), float(square_mm), float(marker_mm), aruco_dict)
    # 여전히 실패 시: 설치/가림 이슈 가능성
    raise RuntimeError("This OpenCV build lacks CharucoBoard API (constructor/create). Ensure opencv-contrib-python is installed and no local cv2.* shadows.")



def create_detector(aruco_dict):
    ar = cv2.aruco
    if hasattr(ar, "DetectorParameters"):
        params = ar.DetectorParameters()
        try:
            params.cornerRefinementMethod = getattr(ar, "CORNER_REFINE_SUBPIX", 1)
        except Exception:
            pass
        detector = ar.ArucoDetector(aruco_dict, params)
        return ("new", detector, params)
    else:
        params = ar.DetectorParameters_create()
        try:
            params.cornerRefinementMethod = getattr(ar, "CORNER_REFINE_SUBPIX", 1)
        except Exception:
            pass
        return ("old", params, params)


def detect_markers(gray, aruco_dict, det_kind, det_obj):
    ar = cv2.aruco
    if det_kind == "new":
        corners, ids, rejected = det_obj.detectMarkers(gray)
    else:
        corners, ids, rejected = ar.detectMarkers(gray, aruco_dict, parameters=det_obj)
    return corners, ids, rejected


def refine_markers(gray, board, corners, ids, rejected):
    ar = cv2.aruco
    if hasattr(ar, "refineDetectedMarkers"):
        try:
            ar.refineDetectedMarkers(gray, board, corners, ids, rejected, cameraMatrix=None, distCoeffs=None)
        except Exception:
            pass
    return corners, ids, rejected


def interpolate_charuco(gray, corners, ids, board, cameraMatrix=None, distCoeffs=None):
    """OpenCV 버전별 반환값 호환 처리
    - 구버전: (charucoCorners, charucoIds)
    - 신버전(일부 4.10+): (retval, charucoCorners, charucoIds)
    입력 마커가 없으면 (None, None) 반환
    """
    ar = cv2.aruco
    if ids is None or corners is None or len(corners) == 0:
        return None, None
    ret = ar.interpolateCornersCharuco(corners, ids, gray, board, cameraMatrix, distCoeffs)
    if isinstance(ret, tuple):
        if len(ret) == 2:
            c_corners, c_ids = ret
        elif len(ret) == 3:
            _, c_corners, c_ids = ret
        else:
            c_corners, c_ids = ret[-2], ret[-1]
    else:
        c_corners, c_ids = ret, None
    return c_corners, c_ids


def calibrate_charuco_single(all_charuco_corners, all_charuco_ids, board, img_size):
    ar = cv2.aruco
    flags = cv2.CALIB_RATIONAL_MODEL
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    if hasattr(ar, "calibrateCameraCharucoExtended"):
        rms, K, D, rvecs, tvecs, *_ = ar.calibrateCameraCharucoExtended(
            all_charuco_corners, all_charuco_ids, board, img_size, None, None,
            flags=flags, criteria=criteria)
        return rms, K, D, rvecs, tvecs
    else:
        rms, K, D, rvecs, tvecs = ar.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board, img_size, None, None,
            flags=flags, criteria=criteria)
        return rms, K, D, rvecs, tvecs


def get_board_corners3d(board):
    # Charuco 보드의 각 체스보드 코너 3D 좌표 (단위: 입력한 square 길이와 동일)
    if hasattr(board, "chessboardCorners"):
        return np.array(board.chessboardCorners, dtype=np.float32)
    if hasattr(board, "getChessboardCorners"):
        return np.array(board.getChessboardCorners(), dtype=np.float32)
    raise RuntimeError("Unable to access board chessboardCorners.")


def match_charuco_pairs(L_cc, L_ids, R_cc, R_ids, board):
    # 각 페어에서 공통 ID만 사용하여 좌/우 2D 점과 3D 보드 점을 구성
    obj_pts, img1_pts, img2_pts = [], [], []
    for lc, li, rc, ri in zip(L_cc, L_ids, R_cc, R_ids):
        li = li.flatten().astype(int); ri = ri.flatten().astype(int)
        common = np.intersect1d(li, ri)
        if len(common) < 12:  # 최소 코너 수 조건
            obj_pts.append(None); img1_pts.append(None); img2_pts.append(None)
            continue
        li_map = {id_: idx for idx, id_ in enumerate(li)}
        ri_map = {id_: idx for idx, id_ in enumerate(ri)}
        corners3d = get_board_corners3d(board)
        obj = corners3d[common].reshape(-1,3).astype(np.float32)
        lpts = np.array([lc[li_map[i]][0] for i in common], dtype=np.float32)
        rpts = np.array([rc[ri_map[i]][0] for i in common], dtype=np.float32)
        obj_pts.append(obj)
        img1_pts.append(lpts)
        img2_pts.append(rpts)
    filt = [(o,i1,i2) for o,i1,i2 in zip(obj_pts,img1_pts,img2_pts) if o is not None]
    if len(filt) == 0:
        raise RuntimeError("No stereo pairs with sufficient common ChArUco corners.")
    obj_pts, img1_pts, img2_pts = map(list, zip(*filt))
    return obj_pts, img1_pts, img2_pts


def save_yml(path, data: dict):
    path = str(path)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if fs.isOpened():
        for k,v in data.items():
            if isinstance(v, (list, tuple)):
                v = np.array(v)
            if isinstance(v, np.ndarray):
                fs.write(k, v)
            else:
                fs.write(k, json.dumps(v))
        fs.release()
        print(f"[OK] saved {path}")
        return
    # FileStorage 실패 시 NPZ로 백업 저장
    np.savez_compressed(path.replace(".yml",".npz"), **{k:np.array(v) if isinstance(v, (list,tuple)) else v for k,v in data.items()})
    print(f"[OK] saved {path.replace('.yml','.npz')}")

# ---------------------- MVSDK 카메라 래퍼 ----------------------
class Cam:
    """MVSDK 카메라 1대를 다루는 간단 래퍼"""
    def __init__(self, dev_info):
        self.dev = dev_info
        self.h = 0
        self.is_mono = False
        self.pFrameBuffer = 0
        self.FrameBufferSize = 0
        self.name = None
        self._soft = False

    def open(self, exposure_us=30000, soft_trigger=True):
        self._soft = soft_trigger
        # 카메라 초기화
        self.h = _init_with_retry(self.dev)
        self.name = getattr(self.dev, 'GetFriendlyName', lambda: 'Camera')()

        # Capability 읽기 및 모노/컬러 판단
        cap = mvsdk.CameraGetCapability(self.h)
        try:
            self.is_mono = bool(cap.sIspCapacity.bMonoSensor)
        except Exception:
            self.is_mono = ("mono" in self.name.lower())

        # ISP 출력 포맷 설정 (모노: MONO8, 컬러: BGR8)
        mvsdk.CameraSetIspOutFormat(
            self.h,
            mvsdk.CAMERA_MEDIA_TYPE_MONO8 if self.is_mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8,
        )

        # 자동노출 끄고 수동 노출 설정(가능한 경우)
        try:
            mvsdk.CameraSetAeState(self.h, 0)
            mvsdk.CameraSetExposureTime(self.h, int(exposure_us))
        except Exception:
            pass

        # 트리거 모드 설정
        m = 1 if soft_trigger else 0  # 1=소프트 트리거, 0=연속
        mvsdk.CameraSetTriggerMode(self.h, m)

        # 스트리밍 시작
        mvsdk.CameraPlay(self.h)

        # 최대 해상도 기준으로 프레임 버퍼 할당
        wmax = cap.sResolutionRange.iWidthMax
        hmax = cap.sResolutionRange.iHeightMax
        ch = 1 if self.is_mono else 3
        self.FrameBufferSize = int(wmax * hmax * ch)
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(self.FrameBufferSize, 16)

    def close(self):
        if self.h:
            try:
                mvsdk.CameraUnInit(self.h)
            except Exception:
                pass
            self.h = 0
        if self.pFrameBuffer:
            try:
                mvsdk.CameraAlignFree(self.pFrameBuffer)
            except Exception:
                pass
            self.pFrameBuffer = 0

    def _numpy_from_pbuffer(self, FrameHead):
        h = FrameHead.iHeight
        w = FrameHead.iWidth
        c = 1 if self.is_mono else 3
        size = int(w * h * c)
        buf_type = ctypes.c_ubyte * size
        buf = buf_type.from_address(int(self.pFrameBuffer))
        arr = np.frombuffer(buf, dtype=np.uint8)
        if c == 1:
            frame = arr.reshape(h, w)
        else:
            frame = arr.reshape(h, w, 3)
        return frame.copy()

    def snap_once(self, timeout_ms=2000):
        """프레임 1장 취득.
        - 소프트트리거 모드면 트리거 후 대기
        - 타임아웃(-12) 발생 시 소프트트리거 재시도 → 연속모드 폴백
        """
        last_exc = None
        for attempt in range(3):
            try:
                if self._soft:
                    # 소프트 트리거를 두 번 짧게 보내 파이프라인을 깨움
                    mvsdk.CameraSoftTrigger(self.h)
                    time.sleep(0.005)
                    try:
                        mvsdk.CameraSoftTrigger(self.h)
                    except Exception:
                        pass
                pRaw, FrameHead = mvsdk.CameraGetImageBuffer(self.h, int(timeout_ms))
                try:
                    mvsdk.CameraImageProcess(self.h, pRaw, self.pFrameBuffer, FrameHead)
                finally:
                    mvsdk.CameraReleaseImageBuffer(self.h, pRaw)
                frame = self._numpy_from_pbuffer(FrameHead)
                return frame, FrameHead
            except Exception as e:
                s = str(e)
                last_exc = e
                # -12 timeout / "超时" / 인코딩 깨짐("瓒呮椂") / 영어 메시지
                if ("-12" in s) or ("timeout" in s.lower()) or ("超时" in s) or ("瓒呮椂" in s):
                    if attempt == 0:
                        time.sleep(0.02)
                        continue
                    if attempt == 1 and self._soft:
                        try:
                            mvsdk.CameraSetTriggerMode(self.h, 0)  # 연속취득으로 폴백
                            self._soft = False
                            time.sleep(0.05)
                            continue
                        except Exception:
                            pass
                raise
        raise last_exc

# ---------------------- 라이브 캡처 + 캘리브레이션 ----------------------
def draw_info(img, text, org=(10,30)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255) if img.ndim==3 else (200,), 2, cv2.LINE_AA)

# 미리보기용 축소 함수: 원본 크기는 유지하고, 화면에만 축소해 표시
# scale과 max_width를 동시에 주면 둘 중 더 작은 배율을 사용
# 반환: (축소이미지, 사용된 배율)
def make_display(img, scale=None, max_width=None):
    h, w = img.shape[:2]
    s = 1.0
    if max_width and max_width > 0:
        s = min(s, max_width / float(w))
    if scale and 0 < scale < 1.0:
        s = min(s, float(scale))
    if s < 1.0:
        new_size = (max(1, int(w*s)), max(1, int(h*s)))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), s
    return img, 1.0


def main():
    ap = argparse.ArgumentParser(description="MVSDK 2-Cam Live ChArUco Stereo Calibration")
    ap.add_argument("--pairs", type=int, default=20, help="필요한 좌/우 쌍 수")
    ap.add_argument("--dict", default="5X5_1000", help="Aruco 딕셔너리")
    ap.add_argument("--squaresX", type=int, default=10)
    ap.add_argument("--squaresY", type=int, default=12)
    ap.add_argument("--square", type=float, default=20.0, help="정사각 타일 변(mm)")
    ap.add_argument("--marker", type=float, default=14.0, help="마커 변(mm)")
    ap.add_argument("--min-corners", type=int, default=20, help="ChArUco 최소 코너 수")
    ap.add_argument("--exposure-us", type=int, default=30000, help="노출 시간(마이크로초)")
    ap.add_argument("--out", default="out_live")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--disp-scale", type=float, default=0.33, help="미리보기 축소 배율(0~1). 1.0=원본")
    ap.add_argument("--disp-max-width", type=int, default=1280, help="미리보기 가로 최대 픽셀. 0=무제한")
    ap.add_argument("--trigger", choices=["soft","cont"], default="cont", help="소프트 트리거(soft) 또는 연속취득(cont)")
    args = ap.parse_args()

    outdir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    if args.debug:
        (outdir / "debug_left").mkdir(parents=True, exist_ok=True)
        (outdir / "debug_right").mkdir(parents=True, exist_ok=True)

    print("OpenCV:", cv2.__version__)

    # 1) 카메라 열기(2대)
    devs = mvsdk.CameraEnumerateDevice()
    if len(devs) < 2:
        print("[ERR] 2대 이상의 카메라가 필요합니다. Found:", len(devs))
        sys.exit(1)

    camL = Cam(devs[0]); camR = Cam(devs[1])
    try:
        camL.open(exposure_us=args.exposure_us, soft_trigger=(args.trigger=="soft"))
        time.sleep(0.3)
        camR.open(exposure_us=args.exposure_us, soft_trigger=(args.trigger=="soft"))
    except Exception as e:
        print("[ERR] 카메라 오픈 실패:", e)
        try:
            camL.close(); camR.close()
        except Exception:
            pass
        sys.exit(2)

    print(f"LEFT : {camL.name} ({'MONO' if camL.is_mono else 'COLOR'})")
    print(f"RIGHT: {camR.name} ({'MONO' if camR.is_mono else 'COLOR'})")

    # 2) 보드/검출기 준비
    aruco_dict = get_aruco_dict(args.dict)
    board = create_charuco_board(args.squaresX, args.squaresY, args.square, args.marker, aruco_dict)
    det_kind, det_obj, _ = create_detector(aruco_dict)

    # 3) 수집 버퍼
    L_cc, L_ids, R_cc, R_ids = [], [], [], []
    L_imgs, R_imgs = [], []  # 원본 프레임 저장 경로 기록

    # 4) 라이브 미预览 루프
    idx = 0
    swap_lr = False
    try:
        while True:
            # 동시 소프트 트리거 → 프레임 획득
            frameL, headL = camL.snap_once(timeout_ms=4000)
            frameR, headR = camR.snap_once(timeout_ms=2000)

            if swap_lr:
                frameL, frameR = frameR, frameL
                headL, headR = headR, headL

            # 뷰어용 복사
            visL = frameL.copy(); visR = frameR.copy()

            # 그레이 변환(aruco 검출)
            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY) if frameL.ndim==3 else frameL
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY) if frameR.ndim==3 else frameR

            # 마커 검출 & 정제
            cL, idL, rejL = detect_markers(grayL, aruco_dict, det_kind, det_obj)
            cR, idR, rejR = detect_markers(grayR, aruco_dict, det_kind, det_obj)
            cL, idL, rejL = refine_markers(grayL, board, cL, idL, rejL)
            cR, idR, rejR = refine_markers(grayR, board, cR, idR, rejR)

            # Charuco 코너 보간
            ccL, ciL = interpolate_charuco(grayL, cL, idL, board)
            ccR, ciR = interpolate_charuco(grayR, cR, idR, board)

            # 디버그 그리기
            try:
                if idL is not None:
                    cv2.aruco.drawDetectedMarkers(visL, cL, idL)
                if ccL is not None:
                    cv2.aruco.drawDetectedCornersCharuco(visL, ccL, ciL, (0,255,0))
            except Exception:
                pass
            try:
                if idR is not None:
                    cv2.aruco.drawDetectedMarkers(visR, cR, idR)
                if ccR is not None:
                    cv2.aruco.drawDetectedCornersCharuco(visR, ccR, ciR, (0,255,0))
            except Exception:
                pass

            # 정보 텍스트
            draw_info(visL, f"L corners: {0 if ciL is None else len(ciL)}  [SPACE/c: 캡처, s:강제저장, x:스왑, q:종료]")
            draw_info(visR, f"R corners: {0 if ciR is None else len(ciR)}  saved: {len(L_cc)}/{args.pairs}")

            # 표시
            # 미리보기 축소 적용
            dispL, _ = make_display(visL, args.disp_scale, args.disp_max_width)
            dispR, _ = make_display(visR, args.disp_scale, args.disp_max_width)

            cv2.imshow("LEFT", dispL)
            cv2.imshow("RIGHT", dispR)

            # 키 처리
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):  # q or ESC
                break
            if k in (ord('x'),):
                swap_lr = not swap_lr
                print("[INFO] 좌/우 스왑:", swap_lr)
            if k in (ord(' '), ord('c')):
                # 조건: 최소 코너 수 만족하면 채택
                nL = 0 if ciL is None else len(ciL)
                nR = 0 if ciR is None else len(ciR)
                if nL >= args.min_corners and nR >= args.min_corners:
                    L_cc.append(ccL); L_ids.append(ciL)
                    R_cc.append(ccR); R_ids.append(ciR)
                    # 원본 저장
                    pL = outdir/"raw"/f"L_{idx:03d}.png"
                    pR = outdir/"raw"/f"R_{idx:03d}.png"
                    cv2.imwrite(str(pL), grayL)
                    cv2.imwrite(str(pR), grayR)
                    L_imgs.append(str(pL)); R_imgs.append(str(pR))
                    print(f"[OK] 캡처 {idx:03d}: corners L={nL} R={nR}")
                    # 디버그 저장
                    if args.debug:
                        dvL = outdir/"debug_left"/f"detect_{idx:03d}.png"
                        dvR = outdir/"debug_right"/f"detect_{idx:03d}.png"
                        cv2.imwrite(str(dvL), visL)
                        cv2.imwrite(str(dvR), visR)
                    idx += 1
                else:
                    print(f"[WARN] 코너 부족: L={nL} R={nR} (min={args.min_corners})")
            if k in (ord('s'),):
                # 강제 저장(검출 불충분해도 원본만 저장)
                pL = outdir/"raw"/f"L_FORCED_{idx:03d}.png"
                pR = outdir/"raw"/f"R_FORCED_{idx:03d}.png"
                cv2.imwrite(str(pL), grayL)
                cv2.imwrite(str(pR), grayR)
                print(f"[SAVE] 강제 원본 저장 {idx:03d}")
                idx += 1

            if len(L_cc) >= args.pairs:
                print("[INFO] 필요한 쌍 수를 달성했습니다. 보정을 시작합니다.")
                break

    finally:
        cv2.destroyAllWindows()

    if len(L_cc) < 2:
        print("[ERR] 유효한 페어가 너무 적습니다.")
        camL.close(); camR.close();
        return

    # 이미지 크기(좌/우 동일 가정: 캡처 시점의 프레임 크기 사용)
    # 캘리브레이션 API는 단일 이미지 크기를 요구하므로 좌/우가 다르면 경고
    img_size = (frameL.shape[1], frameL.shape[0])  # (width, height)
    if (frameR.shape[1], frameR.shape[0]) != img_size:
        print("[WARN] 좌/우 해상도가 다릅니다. 좌측 크기로 진행합니다.")

    # 5) 단안 보정(좌/우)
    print("\n[Calibrating LEFT intrinsics]")
    rmsL, KL, DL, rvecsL, tvecsL = calibrate_charuco_single(L_cc, L_ids, board, img_size)
    print(f"LEFT RMS reprojection: {rmsL:.4f} px")

    print("\n[Calibrating RIGHT intrinsics]")
    rmsR, KR, DR, rvecsR, tvecsR = calibrate_charuco_single(R_cc, R_ids, board, img_size)
    print(f"RIGHT RMS reprojection: {rmsR:.4f} px")

    save_yml(outdir/"intrinsics_left.yml",  {"K":KL, "D":DL, "image_size":np.array(img_size), "rms":np.array([rmsL])})
    save_yml(outdir/"intrinsics_right.yml", {"K":KR, "D":DR, "image_size":np.array(img_size), "rms":np.array([rmsR])})

    # 6) 스테레오 보정(내부 고정)
    print("\n[Matching ChArUco IDs across pairs]")
    obj_pts, img1_pts, img2_pts = match_charuco_pairs(L_cc, L_ids, R_cc, R_ids, board)
    print(f"Stereo usable pairs: {len(obj_pts)}")

    flags = (cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    print("\n[StereoCalibrate]")
    retval, KL2, DL2, KR2, DR2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img1_pts, img2_pts, KL, DL, KR, DR, img_size,
        flags=flags, criteria=criteria)
    print(f"STEREO RMS reprojection: {retval:.4f} px")
    # 보드 단위를 mm로 줬다면 T 역시 mm 단위

    # 7) 스테레오 정렬/리맵 맵 생성
    print("\n[StereoRectify & Rectify maps]")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        KL, DL, KR, DR, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1_L, map2_L = cv2.initUndistortRectifyMap(KL, DL, R1, P1, img_size, cv2.CV_16SC2)
    map1_R, map2_R = cv2.initUndistortRectifyMap(KR, DR, R2, P2, img_size, cv2.CV_16SC2)

    save_yml(outdir/"stereo.yml", {
        "R":R, "T":T, "E":E, "F":F, "R1":R1, "R2":R2, "P1":P1, "P2":P2, "Q":Q,
        "roi1":np.array(roi1), "roi2":np.array(roi2),
        "rms":np.array([retval])
    })
    np.savez_compressed(outdir/"rectify_maps_left.npz",  map1=map1_L, map2=map2_L)
    np.savez_compressed(outdir/"rectify_maps_right.npz", map1=map1_R, map2=map2_R)
    print("[OK] wrote rectify maps npz")

    # 8) 정렬 검증용 시각화(첫 페어)
    try:
        if len(L_imgs) > 0 and len(R_imgs) > 0:
            l0 = cv2.imread(L_imgs[0], cv2.IMREAD_GRAYSCALE)
            r0 = cv2.imread(R_imgs[0], cv2.IMREAD_GRAYSCALE)
            l_rect = cv2.remap(l0, map1_L, map2_L, cv2.INTER_LINEAR)
            r_rect = cv2.remap(r0, map1_R, map2_R, cv2.INTER_LINEAR)
            vis = np.hstack([l_rect, r_rect])
            h, w = vis.shape[:2]
            for yy in np.linspace(20, h-20, 12).astype(int):
                cv2.line(vis, (0,yy), (w-1,yy), (255,), 1, lineType=cv2.LINE_AA)
            cv2.imwrite(str(outdir/"rectified_check.png"), vis)
            print("[OK] wrote rectified_check.png")
    except Exception as e:
        print("[WARN] rectified check image failed:", e)

    # 자원 해제
    camL.close(); camR.close()
    print("\n[DONE] 결과 폴더:", str(outdir))


if __name__ == "__main__":
    main()
