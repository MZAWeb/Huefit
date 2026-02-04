# T-Shirt Color to Hue Lights Sync (with OpenAI API & Room Targeting)
# Description: This script captures video from your webcam,
# uses the OpenAI API to identify the color of your shirt,
# and sets your Philips Hue lights in a specified room to match that color
# at a configurable brightness level.

import cv2
import numpy as np
from phue import Bridge
import os
import sys
import time
import requests
import base64
import ast
import json
import math
import configparser

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hue.conf')
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hue.conf.sample')


def load_config(path=CONFIG_PATH):
    """Load configuration from hue.conf. Exit with a helpful message if missing."""
    if not os.path.exists(path):
        print(f"Configuration file not found: {path}")
        print(f"Copy {SAMPLE_PATH} to {path} and fill in your values.")
        sys.exit(1)
    config = configparser.ConfigParser()
    config.read(path)
    return config


# --- Helper Functions ---

def _convert_rgb_to_xy_bri(r_in, g_in, b_in):
    """
    Converts RGB color to XY color space and brightness for Philips Hue.
    Input R, G, B are 0-255.
    Returns ([x, y], bri) where x, y are 0-1, bri is 1-254.
    Based on Philips Hue developer documentation and phue library implementation.
    """
    # Normalize RGB values to 0-1
    r = r_in / 255.0
    g = g_in / 255.0
    b = b_in / 255.0

    # Apply gamma correction
    r_g = pow((r + 0.055) / (1.0 + 0.055), 2.4) if (r > 0.04045) else (r / 12.92)
    g_g = pow((g + 0.055) / (1.0 + 0.055), 2.4) if (g > 0.04045) else (g / 12.92)
    b_g = pow((b + 0.055) / (1.0 + 0.055), 2.4) if (b > 0.04045) else (b / 12.92)

    # Convert to XYZ using Philips Hue specified sRGB matrix
    X = r_g * 0.664511 + g_g * 0.154324 + b_g * 0.162028
    Y = r_g * 0.283881 + g_g * 0.668433 + b_g * 0.047685
    Z = r_g * 0.000088 + g_g * 0.072310 + b_g * 0.986039

    # Calculate XY coordinates
    if (X + Y + Z) == 0:
        xy = [0.3127, 0.3290]
    else:
        x_coord = X / (X + Y + Z)
        y_coord = Y / (X + Y + Z)
        xy = [float(x_coord), float(y_coord)]

    # Calculate brightness (bri) from luminance (Y)
    bri = math.ceil(Y * 254.0)
    bri = max(1, bri)
    bri = min(254, bri)

    if r_in == 0 and g_in == 0 and b_in == 0:
        bri = 1

    return xy, bri


def connect_to_hue_bridge(ip_address, username, config):
    """
    Connects to the Philips Hue Bridge.
    If no username is stored, prompts the user to press the link button,
    then saves the resulting username back to hue.conf.
    """
    print(f"Attempting to connect to Hue Bridge at {ip_address}...")
    try:
        if username:
            bridge = Bridge(ip_address, username=username)
        else:
            print("No bridge username found in configuration.")
            print("Please press the link button on your Philips Hue Bridge and then press Enter here within 30 seconds.")
            input("Waiting for you to press Enter after pressing the link button...")
            bridge = Bridge(ip_address)
            bridge.connect()
            # Save the new username back to hue.conf
            new_username = bridge.username
            if new_username:
                config.set('hue', 'bridge_username', new_username)
                with open(CONFIG_PATH, 'w') as f:
                    config.write(f)
                print(f"Bridge username saved to {CONFIG_PATH}.")

        lights_available = bridge.get_light_objects('list')
        print(f"Successfully connected to Hue Bridge at {ip_address}!")
        print(f"Found {len(lights_available)} light(s) in total.")
        return bridge
    except Exception as e:
        print(f"Error connecting to Hue Bridge: {e}")
        print("Please check IP, network, and link button status.")
        return None

def get_dominant_color_from_openai(frame, api_key, model, api_url):
    """
    Sends the frame to OpenAI API to get the dominant shirt color.
    Returns an (R, G, B) tuple or None if an error occurs.
    """
    print(f"Sending image to OpenAI API ({model}) for color analysis...")
    if frame is None or frame.size == 0:
        print("Error: Input frame for OpenAI is empty.")
        return None

    success, encoded_image_bytes = cv2.imencode('.png', frame)
    if not success:
        print("Error: Could not encode frame to PNG.")
        return None
    base64_image_data = base64.b64encode(encoded_image_bytes.tobytes()).decode('utf-8')

    prompt_text = (
        "You are an expert color analyzer and interior lighting designer. Look at the person in the image. "
        "What color would create the most aesthetically pleasing lighting effect that complements their clothing? "
        "Consider the following:\n"
        "- For solid colored garments, use that color\n"
        "- For patterned clothing, choose the most prominent/accent color that would create a nice atmosphere\n"
        "- For white/light clothing with logos or patterns, use the pattern/logo color\n"
        "- For very dark clothing, choose a complementary color that would create a nice contrast\n"
        "Respond ONLY with the RGB color values as a Python-style tuple string. For example:\n"
        "- If it's red, respond: (255, 0, 0)\n"
        "- If it's dark blue, respond: (0, 0, 139)\n"
        "Do not include any other text, explanation, or markdown formatting. "
        "If no clear clothing is visible or if the image is unclear, respond with: (0, 0, 0)"
    )
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text},
                                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_data}"}}]}],
        "max_tokens": 60
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        response_json = response.json()
        if (response_json.get("choices") and len(response_json["choices"]) > 0 and
            response_json["choices"][0].get("message") and response_json["choices"][0]["message"].get("content")):
            color_text = response_json["choices"][0]["message"]["content"].strip()
            print(f"OpenAI API response (color text): '{color_text}'")
            try:
                rgb_color = ast.literal_eval(color_text)
                if isinstance(rgb_color, tuple) and len(rgb_color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in rgb_color):
                    print(f"Successfully parsed RGB color: {rgb_color}")
                    return rgb_color
                else:
                    print(f"Error: OpenAI response '{color_text}' is not a valid RGB tuple. Defaulting to black.")
                    return (0, 0, 0)
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Error parsing color string from OpenAI '{color_text}': {e}. Defaulting to black.")
                return (0, 0, 0)
        else:
            print("Error: Unexpected response structure from OpenAI API.")
            if "error" in response_json: print("API Error details:", response_json["error"])
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenAI API: {e}")
        if 'response' in locals() and response is not None:
            print(f"OpenAI API Response Status Code: {response.status_code}")
            try: print(f"OpenAI API Response Body: {response.text}")
            except Exception: pass
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI API call: {e}")
        return None

def set_hue_lights_color(bridge, rgb_color_tuple, target_room_name, brightness_pct):
    """
    Sets the color of Hue lights in the specified room using XY and Brightness.
    brightness_pct is 0-100 and maps to the Hue API range 1-254.
    """
    if not bridge:
        print("Hue Bridge not connected. Cannot set light color.")
        return
    if rgb_color_tuple is None:
        print("No color provided to set_hue_lights_color.")
        return
    if not target_room_name:
        print("No target Hue Room specified in configuration. Lights not changed.")
        return

    r, g, b = rgb_color_tuple

    xy_coords, _ = _convert_rgb_to_xy_bri(r, g, b)

    brightness = max(1, min(254, round(brightness_pct / 100.0 * 254)))

    print(f"Converted RGB({r},{g},{b}) to XY: {xy_coords}.")
    print(f"Setting brightness to {brightness_pct}% (Hue value: {brightness}).")

    command = {
        'on': True,
        'xy': xy_coords,
        'bri': brightness,
        'transitiontime': 10
    }

    try:
        group_id = bridge.get_group_id_by_name(target_room_name)

        if group_id is not None:
            print(f"Setting lights in room '{target_room_name}' (ID: {group_id}) using command: {command}")
            bridge.set_group(group_id, command)
            print(f"Hue lights color update command sent for room '{target_room_name}'.")
        else:
            print(f"Error: Hue Room '{target_room_name}' not found on the bridge. Lights not changed.")
            print("Please ensure 'room' in hue.conf matches a room in your Hue app.")

    except Exception as e:
        print(f"Error setting Hue lights color for room '{target_room_name}': {e}")
        print("Ensure the room name is correct and the bridge is responsive.")


# --- Main Application Logic ---

def main():
    config = load_config()

    bridge_ip = config.get('hue', 'bridge_ip')
    bridge_username = config.get('hue', 'bridge_username')
    target_room = config.get('hue', 'room')
    brightness_pct = config.getint('hue', 'brightness', fallback=100)
    openai_api_key = config.get('openai', 'api_key')
    openai_model = config.get('openai', 'model')
    openai_api_url = config.get('openai', 'api_url')

    if not bridge_ip:
        print("Error: 'bridge_ip' is not set in hue.conf. Please fill it in.")
        sys.exit(1)
    if not openai_api_key:
        print("Error: 'api_key' is not set in hue.conf. Please fill it in.")
        sys.exit(1)

    print("-----------------------------------------------------------------")
    print(" T-Shirt Color (via OpenAI API) to Hue Lights Sync (Room Mode) ")
    print("-----------------------------------------------------------------")
    print(f"This script uses your webcam and OpenAI API ({openai_model}) to detect your shirt color")
    print(f"and sets Philips Hue lights in the room '{target_room}' accordingly at {brightness_pct}% brightness.")
    print("-----------------------------------------------------------------\n")

    bridge = connect_to_hue_bridge(bridge_ip, bridge_username, config)
    if not bridge: print("Failed to connect to Hue Bridge. Exiting."); return

    if bridge and target_room:
        print(f"\nChecking for target Hue Room: '{target_room}'...")
        try:
            target_room_id = bridge.get_group_id_by_name(target_room)
            if target_room_id is not None:
                group_info = bridge.get_group(target_room_id)
                lights_in_room = group_info.get('lights', [])
                print(f"Successfully found Hue Room: '{target_room}' (ID: {target_room_id}) with {len(lights_in_room)} light(s).")
            else:
                print(f"Warning: Target Hue Room '{target_room}' not found on the bridge.")
                print("Please check the 'room' setting in hue.conf and ensure it matches a room in your Philips Hue app.")
                print("Available rooms/groups found on bridge:")
                all_groups = bridge.get_group()
                room_found = False
                for gid, ginfo in all_groups.items():
                    if ginfo.get('type') in ['Room', 'Zone']:
                        print(f"  - Name: {ginfo['name']} (ID: {gid}, Type: {ginfo.get('type')})")
                        room_found = True
                if not room_found:
                    print("  No rooms/zones found on the bridge. Check your Hue app setup.")
                print("The script will attempt to proceed, but light changes will fail if the room is not correctly identified.")
        except Exception as e:
            print(f"Error when checking for Hue Room '{target_room}': {e}")
            print("Proceeding, but light changes may fail.")
    elif not target_room:
        print("\nWarning: 'room' is not set in hue.conf. Light changes will not be targeted to a specific room.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: Could not open webcam."); return

    print("\nWebcam opened successfully.")
    print("Instructions:")
    print(" - Ensure your upper body clothing is clearly visible in the webcam feed.")
    print(f" - Press 'c' to capture color and set Hue lights in room '{target_room}' (can take a few seconds).")
    print(" - Press 'q' to quit.")
    print("-----------------------------------------------------------------\n")

    show_processing_message = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None: print("Error: Can't receive frame. Exiting ..."); break

        display_frame = frame.copy()
        display_frame = cv2.flip(display_frame, 1)

        if show_processing_message:
            cv2.putText(display_frame, f"Processing with OpenAI ({openai_model})...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(f'Webcam - Shirt Color Detector (Room: {target_room})', display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('c'):
            print(f"\nCapturing color with OpenAI API ({openai_model})...")
            show_processing_message = True
            temp_display_frame = display_frame.copy()
            cv2.putText(temp_display_frame, f"Processing with OpenAI ({openai_model})...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(f'Webcam - Shirt Color Detector (Room: {target_room})', temp_display_frame)
            cv2.waitKey(1)

            detected_rgb_color = get_dominant_color_from_openai(frame, openai_api_key, openai_model, openai_api_url)
            show_processing_message = False

            if detected_rgb_color:
                set_hue_lights_color(bridge, detected_rgb_color, target_room, brightness_pct)
            else:
                print("Could not determine color using OpenAI API. Try again or check API key/quota.")

        elif key == ord('q'):
            print("Quitting application..."); break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()
