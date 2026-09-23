#!/usr/bin/env python3
"""Fisheye calibration tool for the 160-degree USB AR0234 camera.

Shows a live view of /dev/video21.  Hold a printed chessboard at many angles
and distances (covering the WHOLE field of view, especially the edges) and
press SPACE to capture each pose.  Aim for 20-30 good poses.  Press 'c' to
run cv2.fisheye.calibrate and write fisheye_calib.json (K + D).  Press 'q' to
quit.

Chessboard default: 8x5 INTERNAL corners (a 9x6 grid of squares).  Print a
flat, rigid board (glue it to a panel) so the squares stay planar.
"""
import argparse, json, time
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video21")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cols", type=int, default=8,
                        help="internal corners along X (cols-1 squares)")
    parser.add_argument("--rows", type=int, default=5,
                        help="internal corners along Y (rows-1 squares)")
    parser.add_argument("--square", type=float, default=1.0,
                        help="square side length in arbitrary units (scale cancels in K)")
    parser.add_argument("--out", default="fisheye_calib.json")
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = (args.cols, args.rows)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("cannot open camera: {}".format(args.camera))

    # Object points: a flat chessboard, square = args.square units.
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    objpoints, imgpoints = [], []
    last_frame = None
    poses = 0

    print("Hold the chessboard at varied angles/distances. SPACE=capture, c=calibrate, q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))

        found, corners = cv2.findChessboardCorners(gray, pattern,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
        if found:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, pattern, corners * (display.shape[1] / frame.shape[1]), found)

        cv2.putText(display, "poses: {}  SPACE=capture c=calibrate q=quit".format(poses),
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("fisheye calibrate", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" ") and found:
            objpoints.append(objp)
            imgpoints.append(corners)
            poses += 1
            print("captured pose {}".format(poses))
        elif key == ord("c"):
            if poses < 10:
                print("need >=10 poses, have {}".format(poses))
                continue
            print("calibrating on {} poses ...".format(poses))
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                     | cv2.fisheye.CALIB_CHECK_COND
                     | cv2.fisheye.CALIB_FIX_SKEW)
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, imgpoints, (args.width, args.height), K, D, flags=flags, criteria=criteria)
            data = {
                "K": K.tolist(),
                "D": D.reshape(-1).tolist(),
                "size": [args.width, args.height],
                "square": args.square,
                "rms_px": float(rms),
                "n_poses": poses,
            }
            with open(args.out, "w") as handle:
                json.dump(data, handle, indent=2)
            print("saved {}  rms={:.3f}px  fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}".format(
                args.out, rms, K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
            # show a quick undistorted preview
            dim = (args.width, args.height)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=0.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            small = cv2.resize(undistorted, (960, int(960 * undistorted.shape[0] / undistorted.shape[1])))
            cv2.imshow("undistorted preview (press any key)", small)
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
