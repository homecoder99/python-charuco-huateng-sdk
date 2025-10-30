# stereo_charuco_calib.py
# deps: opencv-contrib-python>=4.7, numpy, pillow (디버그 저장용), (선택)reportlab
import os, glob, argparse, json
from pathlib import Path
import numpy as np
import cv2

# ---------- Helpers (OpenCV 버전 호환) ----------
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
    # getPredefinedDictionary vs Dictionary_get
    if hasattr(ar, "getPredefinedDictionary"):
        return ar.getPredefinedDictionary(dict_id)
    return ar.Dictionary_get(dict_id)

def create_charuco_board(squaresX, squaresY, square_mm, marker_mm, aruco_dict):
    ar = cv2.aruco
    # OpenCV 4.7+: CharucoBoard((cols, rows), square, marker, dict)
    if hasattr(ar, "CharucoBoard") and hasattr(ar.CharucoBoard, "create"):
        return ar.CharucoBoard.create((squaresX, squaresY), float(square_mm), float(marker_mm), aruco_dict)
    # 구버전: CharucoBoard_create(cols, rows, square, marker, dict)
    if hasattr(ar, "CharucoBoard_create"):
        return ar.CharucoBoard_create(squaresX, squaresY, float(square_mm), float(marker_mm), aruco_dict)
    raise RuntimeError("This OpenCV build lacks CharucoBoard. Install opencv-contrib-python.")

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
    ar = cv2.aruco
    # interpolateCornersCharuco(gray? No: signature uses (markerCorners, markerIds, image, board, ...)
    return ar.interpolateCornersCharuco(corners, ids, gray, board, cameraMatrix, distCoeffs)

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
    # board.chessboardCorners is Nx3 float32 (same 단위: 여기서는 mm로 입력 권장)
    if hasattr(board, "chessboardCorners"):
        return np.array(board.chessboardCorners, dtype=np.float32)
    # fallbacks (rare)
    if hasattr(board, "getChessboardCorners"):
        return np.array(board.getChessboardCorners(), dtype=np.float32)
    raise RuntimeError("Unable to access board chessboardCorners.")

# ---------- Core pipeline ----------
def load_pairs(left_dir, right_dir, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")):
    L = []
    R = []
    for e in exts:
        L += sorted(glob.glob(str(Path(left_dir) / e)))
        R += sorted(glob.glob(str(Path(right_dir) / e)))
    # 간단히 인덱스 매칭(동일 개수/순서 가정)
    n = min(len(L), len(R))
    if n == 0:
        raise RuntimeError("No images found. Check folders and extensions.")
    return L[:n], R[:n]

def detect_charuco_on_images(img_paths, aruco_dict, board, min_corners=20, debug_dir=None):
    kind, det, _ = create_detector(aruco_dict)
    all_cc, all_ids, keep_idx, dbg = [], [], [], []
    Ht, Wd = None, None
    for i, p in enumerate(img_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] cannot read: {p}")
            continue
        Ht, Wd = img.shape[:2]
        corners, ids, rejected = detect_markers(img, aruco_dict, kind, det)
        corners, ids, rejected = refine_markers(img, board, corners, ids, rejected)
        if ids is None or len(ids) == 0:
            print(f"[WARN] no markers: {p}")
            continue
        # Charuco 서브픽셀 코너
        c_corners, c_ids = interpolate_charuco(img, corners, ids, board)
        if c_ids is None or len(c_ids) < min_corners:
            print(f"[WARN] few ChArUco corners ({0 if c_ids is None else len(c_ids)}): {p}")
            continue
        all_cc.append(c_corners)  # (N,1,2)
        all_ids.append(c_ids)     # (N,1)
        keep_idx.append(i)

        if debug_dir:
            dbg_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(dbg_img, corners, ids)
            try:
                cv2.aruco.drawDetectedCornersCharuco(dbg_img, c_corners, c_ids, (0,255,0))
            except Exception:
                pass
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            outp = str(Path(debug_dir) / f"detect_{i:04d}.png")
            cv2.imwrite(outp, dbg_img)
            dbg.append(outp)

    if len(all_cc) == 0:
        raise RuntimeError("No valid ChArUco detections. Check board/lighting/focus.")
    return all_cc, all_ids, keep_idx, (Wd, Ht), dbg

def match_charuco_pairs(L_cc, L_ids, R_cc, R_ids, board):
    # 각 페어에서 공통 ID만 사용
    obj_pts, img1_pts, img2_pts = [], [], []
    for lc, li, rc, ri in zip(L_cc, L_ids, R_cc, R_ids):
        li = li.flatten().astype(int); ri = ri.flatten().astype(int)
        common = np.intersect1d(li, ri)
        if len(common) < 12:
            obj_pts.append(None); img1_pts.append(None); img2_pts.append(None)
            continue
        # id -> index 매핑
        li_map = {id_: idx for idx, id_ in enumerate(li)}
        ri_map = {id_: idx for idx, id_ in enumerate(ri)}
        # 3D object points (보드 좌표, z=0, 단위=board에서 준 단위: mm 권장)
        corners3d = get_board_corners3d(board)
        obj = corners3d[common].reshape(-1,3).astype(np.float32)
        # 2D image points
        lpts = np.array([lc[li_map[i]][0] for i in common], dtype=np.float32)  # (N,2)
        rpts = np.array([rc[ri_map[i]][0] for i in common], dtype=np.float32)
        obj_pts.append(obj)
        img1_pts.append(lpts)
        img2_pts.append(rpts)
    # 유효 페어만 필터
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
    # fallback: npz
    np.savez_compressed(path.replace(".yml",".npz"), **{k:np.array(v) if isinstance(v, (list,tuple)) else v for k,v in data.items()})
    print(f"[OK] saved {path.replace('.yml','.npz')}")

def main():
    ap = argparse.ArgumentParser(description="Stereo calibration with ChArUco")
    ap.add_argument("--left", required=True, help="left image folder")
    ap.add_argument("--right", required=True, help="right image folder")
    ap.add_argument("--dict", default="5X5_1000", help="Aruco dictionary (e.g., 5X5_1000, 4X4_50)")
    ap.add_argument("--squaresX", type=int, default=10)
    ap.add_argument("--squaresY", type=int, default=12)
    ap.add_argument("--square", type=float, default=20.0, help="square length (mm)")
    ap.add_argument("--marker", type=float, default=14.0, help="marker length (mm)")
    ap.add_argument("--min-corners", type=int, default=20)
    ap.add_argument("--out", default="out")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    print("OpenCV:", cv2.__version__)

    aruco_dict = get_aruco_dict(args.dict)
    board = create_charuco_board(args.squaresX, args.squaresY, args.square, args.marker, aruco_dict)

    L_paths, R_paths = load_pairs(args.left, args.right)
    print(f"Found {len(L_paths)} pairs.")

    # Detect ChArUco corners for each camera
    L_cc, L_ids, L_keep, imgsize, L_dbg = detect_charuco_on_images(
        L_paths, aruco_dict, board, min_corners=args.min_corners,
        debug_dir=str(outdir/"debug_left") if args.debug else None
    )
    R_cc, R_ids, R_keep, imgsizeR, R_dbg = detect_charuco_on_images(
        R_paths, aruco_dict, board, min_corners=args.min_corners,
        debug_dir=str(outdir/"debug_right") if args.debug else None
    )

    if imgsize != imgsizeR:
        print("[WARN] left/right image sizes differ. Using left size.")
    img_size = imgsize

    # Single-camera calibration
    print("\n[Calibrating LEFT intrinsics]")
    rmsL, KL, DL, rvecsL, tvecsL = calibrate_charuco_single(L_cc, L_ids, board, img_size)
    print(f"LEFT RMS reprojection: {rmsL:.4f} px")

    print("\n[Calibrating RIGHT intrinsics]")
    rmsR, KR, DR, rvecsR, tvecsR = calibrate_charuco_single(R_cc, R_ids, board, img_size)
    print(f"RIGHT RMS reprojection: {rmsR:.4f} px")

    save_yml(outdir/"intrinsics_left.yml",  {"K":KL, "D":DL, "image_size":np.array(img_size), "rms":np.array([rmsL])})
    save_yml(outdir/"intrinsics_right.yml", {"K":KR, "D":DR, "image_size":np.array(img_size), "rms":np.array([rmsR])})

    # Stereo calibration with fixed intrinsics
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
    # (참고) 보드에 mm를 줬다면 T 단위도 mm입니다.

    # Rectification & maps
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

    # Save rectify maps as npz for 실시간 remap()
    np.savez_compressed(outdir/"rectify_maps_left.npz",  map1=map1_L, map2=map2_L)
    np.savez_compressed(outdir/"rectify_maps_right.npz", map1=map1_R, map2=map2_R)
    print("[OK] wrote rectify maps npz")

    # Debug: 첫 번째 페어 정렬 확인 이미지
    try:
        import math
        l0 = cv2.imread(L_paths[L_keep[0]], cv2.IMREAD_GRAYSCALE)
        r0 = cv2.imread(R_paths[R_keep[0]], cv2.IMREAD_GRAYSCALE)
        l_rect = cv2.remap(l0, map1_L, map2_L, cv2.INTER_LINEAR)
        r_rect = cv2.remap(r0, map1_R, map2_R, cv2.INTER_LINEAR)
        # 에피폴라 확인용 수평선
        vis = np.hstack([l_rect, r_rect])
        h, w = vis.shape[:2]
        for yy in np.linspace(20, h-20, 12).astype(int):
            cv2.line(vis, (0,yy), (w-1,yy), (255,), 1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(outdir/"rectified_check.png"), vis)
        print("[OK] wrote rectified_check.png")
    except Exception as e:
        print(f"[WARN] rectified check image failed: {e}")

if __name__ == "__main__":
    main()
