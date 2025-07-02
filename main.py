import asyncio
import itertools
import os
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Self

import cv2
import mediapipe as mp
import numpy as np
import simpleobsws
from deepface import DeepFace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="face_landmarker_v2_with_blendshapes.task"
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

LEFT_THRESHOLD = 0.21
RIGHT_THRESHOLD = 0.21

STABILITY_MAJORITY = 5
DETECTION_TOTAL_DURATION = 0.01

RAGEY_THRESHOLD = 30


class Emotion(Enum):
    """
    Emotions mapping
    """

    ANGRY = 0
    DISGUSTED = 1
    FEARFUL = 2
    HAPPY = 3
    NEUTRAL = 4
    OTHER = 5
    SAD = 6
    SURPRISED = 7
    UNKNOWN = 8

    @staticmethod
    def from_index(number: int) -> Self:
        return [
            Emotion.ANGRY,
            Emotion.DISGUSTED,
            Emotion.FEARFUL,
            Emotion.HAPPY,
            Emotion.NEUTRAL,
            Emotion.OTHER,
            Emotion.SAD,
            Emotion.SURPRISED,
            Emotion.UNKNOWN,
        ][number]

    @staticmethod
    def from_string(text: str) -> Self:
        match text:
            case "angry":
                return Emotion.ANGRY
            case "disgusted":
                return Emotion.DISGUSTED
            case "fearful":
                return Emotion.FEARFUL
            case "happy":
                return Emotion.HAPPY
            case "neutral":
                return Emotion.NEUTRAL
            case "sad":
                return Emotion.SAD
            case "surprised":
                return Emotion.SURPRISED

        return Emotion.NEUTRAL


# Scene collection for each face
@dataclass
class SceneCollection:
    """
    Scene Collection
    """

    base: str
    leftEye: str
    leftEyeClosed: str
    rightEye: str
    rightEyeClosed: str


SCENE_COLLECTION_ANGRY = SceneCollection(
    "VanorBodyMad",
    "VanorLeftEyeAnger",
    "VanorLeftEyeClosed",
    "VanorRightEyeAnger",
    "VanorRightEyeClosed",
)

SCENE_COLLECTION_D = SceneCollection(
    "VanorBodyMad",
    "VanorLeftEyeOpen",
    "VanorLeftEyeClosed",
    "VanorRightEyeOpen",
    "VanorRightEyeClosed",
)

SCENE_COLLECTION_NEUTRAL = SceneCollection(
    "VanorBody",
    "VanorLeftEyeOpen",
    "VanorLeftEyeClosed",
    "VanorRightEyeOpen",
    "VanorRightEyeClosed",
)

# Scene mapping activation
SCENE_MAPPING = {
    Emotion.ANGRY: SCENE_COLLECTION_ANGRY,
    Emotion.DISGUSTED: SCENE_COLLECTION_ANGRY,
    Emotion.FEARFUL: SCENE_COLLECTION_D,
    Emotion.HAPPY: SCENE_COLLECTION_NEUTRAL,
    Emotion.NEUTRAL: SCENE_COLLECTION_NEUTRAL,
    Emotion.OTHER: SCENE_COLLECTION_NEUTRAL,
    Emotion.SAD: SCENE_COLLECTION_D,
    Emotion.SURPRISED: SCENE_COLLECTION_D,
    Emotion.UNKNOWN: SCENE_COLLECTION_NEUTRAL,
}

# All managed scenes
MANAGED_SCENES = [SCENE_COLLECTION_ANGRY, SCENE_COLLECTION_D, SCENE_COLLECTION_NEUTRAL]


async def update_managed_scene_with_emotion(
    ws: simpleobsws.WebSocketClient, emotion: Emotion, leftEye: bool, rightEye: bool
):
    request = simpleobsws.Request("GetSceneList")
    response = await ws.call(request)
    if not response.ok():
        print("Could not get scene list while updating emotions")

    uuids = [scene["sceneUuid"] for scene in response.responseData["scenes"]]
    target_scene = SCENE_MAPPING[emotion]
    to_enable = [
        target_scene.base,
        target_scene.leftEye if leftEye else target_scene.leftEyeClosed,
        target_scene.rightEye if rightEye else target_scene.rightEyeClosed,
    ]
    possible_scenes = set(
        itertools.chain(
            *map(
                lambda x: [
                    x.base,
                    x.leftEye,
                    x.leftEyeClosed,
                    x.rightEye,
                    x.rightEyeClosed,
                ],
                MANAGED_SCENES,
            )
        )
    )
    for uuid in uuids:
        request = simpleobsws.Request("GetSceneItemList", {"sceneUuid": uuid})
        response = await ws.call(request)
        if not response.ok():
            print(f"Could not get scene item list of {uuid}")
            continue

        sceneItems = response.responseData["sceneItems"]
        for sceneItem in sceneItems:
            sourceName = sceneItem["sourceName"]
            sourceId = sceneItem["sceneItemId"]

            if sourceName not in possible_scenes:
                continue

            request = simpleobsws.Request(
                "SetSceneItemEnabled",
                {
                    "sceneUuid": uuid,
                    "sceneItemId": sourceId,
                    "sceneItemEnabled": sourceName in to_enable,
                },
            )
            response = await ws.call(request)
            if not response.ok():
                print(f"Could not toggle scene item {sourceName}")


def compute_ear(eye: np.ndarray) -> float:
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    ear = (a + b) / (2.0 * c)
    return ear


def get_eye_state(image: np.ndarray) -> tuple[bool, bool]:
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(img)
    # print(len(detection_result.face_landmarks))
    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        h, w, _ = image.shape
        left_eye = np.array(
            [
                [
                    int(face_landmarks[i].x * w),
                    int(face_landmarks[i].y * h),
                ]
                for i in LEFT_EYE_IDX
            ]
        )
        right_eye = np.array(
            [
                [
                    int(face_landmarks[i].x * w),
                    int(face_landmarks[i].y * h),
                ]
                for i in RIGHT_EYE_IDX
            ]
        )

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)

        print(f"current ratios: {left_ear}, {right_ear}")

        return left_ear < LEFT_THRESHOLD, right_ear < RIGHT_THRESHOLD
    return False, False


def get_custom_dominant_emotion(emotion_dict: dict[str]) -> str:
    if emotion_dict["angry"] > RAGEY_THRESHOLD:
        return "angry"

    return max(emotion_dict.items(), key=lambda x: x[1])[0]


def get_emotion(image: np.ndarray) -> Emotion:
    result = DeepFace.analyze(
        image,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
    )

    # emotion = result[0]["dominant_emotion"]
    emotion = get_custom_dominant_emotion(result[0]["emotion"])
    return Emotion.from_string(emotion)


async def main():
    print("Connecting to OBS...")
    password = os.getenv("OBS_PASSWORD")
    ws = simpleobsws.WebSocketClient(url="ws://localhost:4455", password=password)
    await ws.connect()
    await ws.wait_until_identified()

    while True:
        try:
            cap = cv2.VideoCapture(0)
            history = []

            while cap.isOpened():
                start_time = time.time_ns()
                success, img = cap.read()
                if not success:
                    print("Failed to get capture")
                    return

                img = cv2.resize(img, (640, 480))

                emotion = get_emotion(img)
                # left_eye, right_eye = get_eye_state(img)

                if len(history) > STABILITY_MAJORITY:
                    history.pop(0)
                history.append((emotion, False, False))

                # use the majority
                majority_emotion = Counter(map(lambda x: x[0], history)).most_common(1)[
                    0
                ][0]
                majority_left_eye = Counter(map(lambda x: x[1], history)).most_common(
                    1
                )[0][0]
                majority_right_eye = Counter(map(lambda x: x[2], history)).most_common(
                    1
                )[0][0]

                # print(
                #     f"Guessed {majority_emotion}, left eye: {majority_left_eye}, right eye: {majority_right_eye}"
                # )
                # print(f"Guessed {majority_emotion}")
                await update_managed_scene_with_emotion(
                    ws, majority_emotion, not majority_left_eye, not majority_right_eye
                )
                end_time = time.time_ns()

                sleep_duration = (DETECTION_TOTAL_DURATION * 10e9) - (
                    end_time - start_time
                )
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration * 10e-9)

        except KeyboardInterrupt:
            print("Killing script")
        finally:
            await ws.disconnect()
            print("Disconnected from OBS")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
