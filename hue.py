# T-Shirt Color to Hue Lights Sync (with OpenAI API & Room Targeting)
# Description: This script captures video from your webcam,
# uses the OpenAI API (GPT-4o) to identify the color of your shirt,
# and sets your Philips Hue lights in a SPECIFIED ROOM to match that color
# AT MAXIMUM BRIGHTNESS.

# Prerequisites:
# 1. pip install opencv-python numpy phue requests
# 2. Know your Philips Hue Bridge IP address.
# 3. Have an OpenAI API Key with access to GPT-4o (or a similar vision-capable model).
# 4. Be prepared to press the link button on your Hue Bridge the first time.
# 5. Ensure the TARGET_HUE_ROOM_NAME below matches a room in your Hue setup.

import cv2
import numpy as np
from phue import Bridge
import os
import time
import requests # For OpenAI API calls
import base64   # For encoding image to send to OpenAI
import ast      # For safely evaluating string tuple from OpenAI response
import json     # For handling API response
import math     # For color conversion

# --- Configuration ---
# Hue Bridge IP
HUE_BRIDGE_IP = '' # Leave empty or as 'YOUR_HUE_BRIDGE_IP' to be prompted
USER_IP_CONFIG_FILE = 'hue_ip_config.txt'
PHUE_CONFIG_FILE = 'phue_bridge.conf' # For phue library

# OpenAI API Key
OPENAI_API_KEY = '' # Leave empty to be prompted, or set your key here.
USER_OPENAI_KEY_CONFIG_FILE = 'openai_api_key.txt'
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5.2"

# !!! NEW: Specify the Hue Room to control !!!
TARGET_HUE_ROOM_NAME = "Office" # Change this to your desired Hue Room name


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
    # (approximates D65 illuminant)
    X = r_g * 0.664511 + g_g * 0.154324 + b_g * 0.162028
    Y = r_g * 0.283881 + g_g * 0.668433 + b_g * 0.047685 # Y is luminance
    Z = r_g * 0.000088 + g_g * 0.072310 + b_g * 0.986039

    # Calculate XY coordinates
    if (X + Y + Z) == 0:
        # Default to a point in the center of the gamut for black
        xy = [0.3127, 0.3290] # A common white point for "black"
    else:
        x_coord = X / (X + Y + Z)
        y_coord = Y / (X + Y + Z)
        xy = [float(x_coord), float(y_coord)]

    # Calculate brightness (bri) from luminance (Y)
    # Y is 0-1. Brightness for Hue API is 1-254.
    # This calculated brightness will be overridden later if max brightness is desired.
    bri = math.ceil(Y * 254.0) 
    bri = max(1, bri)   # Ensure brightness is at least 1 if light is on
    bri = min(254, bri) # Ensure brightness does not exceed 254

    if r_in == 0 and g_in == 0 and b_in == 0:
        bri = 1 # For pure black, calculated brightness is 1 (dimmest 'on')
        # xy is already set to a white point for black.

    return xy, bri


def get_config_value(config_file, prompt_message, sensitive=False):
    """
    Gets a configuration value from a file or prompts the user.
    Saves the value to the config file for future use.
    """
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            value = f.read().strip()
            if value:
                print(f"Found configuration in '{config_file}'.")
                return value
    
    while True:
        value = input(prompt_message).strip()
        if value:
            if "IP address" in prompt_message: # Basic IP format check
                try:
                    parts = value.split('.')
                    if len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts):
                        pass 
                    else:
                        print("Invalid IP address format. Please try again.")
                        continue
                except ValueError:
                    print("Invalid IP address format. Please enter numbers separated by dots.")
                    continue
            
            with open(config_file, 'w') as f:
                f.write(value)
            print(f"Saved configuration to '{config_file}'.")
            return value
        else:
            print("Value cannot be empty.")

def connect_to_hue_bridge(ip_address):
    """
    Connects to the Philips Hue Bridge.
    """
    print(f"Attempting to connect to Hue Bridge at {ip_address}...")
    try:
        bridge = Bridge(ip_address, config_file_path=PHUE_CONFIG_FILE)
        if not bridge.username:
            print("This appears to be the first connection or username is missing.")
            print(f"Please press the link button on your Philips Hue Bridge and then press Enter here within 30 seconds.")
            input("Waiting for you to press Enter after pressing the link button...")
            bridge.connect()
            print("Attempted to register with the bridge.")
        
        lights_available = bridge.get_light_objects('list')
        print(f"Successfully connected to Hue Bridge at {ip_address}!")
        print(f"Found {len(lights_available)} light(s) in total.")
        return bridge
    except Exception as e:
        print(f"Error connecting to Hue Bridge: {e}")
        print("Please check IP, network, and link button status.")
        return None

def get_dominant_color_from_openai(frame, api_key):
    """
    Sends the frame to OpenAI API (GPT-4o) to get the dominant shirt color.
    Returns an (R, G, B) tuple or None if an error occurs.
    """
    print(f"Sending image to OpenAI API ({OPENAI_MODEL}) for color analysis...")
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
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text},
                                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_data}"}}]}],
        "max_tokens": 60
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=45)
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

def set_hue_lights_color(bridge, rgb_color_tuple, target_room_name):
    """
    Sets the color of Hue lights in the specified room using XY and Brightness.
    Brightness will be forced to maximum (254).
    """
    if not bridge:
        print("Hue Bridge not connected. Cannot set light color.")
        return
    if rgb_color_tuple is None:
        print("No color provided to set_hue_lights_color.")
        return
    if not target_room_name:
        print("No target Hue Room specified in the script configuration (TARGET_HUE_ROOM_NAME). Lights not changed.")
        return

    r, g, b = rgb_color_tuple
    
    # Convert RGB to XY and Brightness
    xy_coords, original_calculated_brightness = _convert_rgb_to_xy_bri(r, g, b)

    # Override brightness to maximum
    brightness = 254 

    print(f"Converted RGB({r},{g},{b}) to XY: {xy_coords}.")
    print(f"Original calculated brightness was: {original_calculated_brightness}. Setting to MAX brightness: {brightness}.")


    command = {
        'on': True,
        'xy': xy_coords,
        'bri': brightness, # Using the overridden max brightness
        'transitiontime': 10  # 1 second transition
    }

    try:
        group_id = bridge.get_group_id_by_name(target_room_name)
        
        if group_id is not None:
            print(f"Setting lights in room '{target_room_name}' (ID: {group_id}) using command: {command}")
            bridge.set_group(group_id, command)
            print(f"Hue lights color update command sent for room '{target_room_name}'.")
        else:
            print(f"Error: Hue Room '{target_room_name}' not found on the bridge. Lights not changed.")
            print("Please ensure TARGET_HUE_ROOM_NAME in the script matches a room in your Hue app.")

    except Exception as e: 
        print(f"Error setting Hue lights color for room '{target_room_name}': {e}")
        print("Ensure the room name is correct and the bridge is responsive.")


# --- Main Application Logic ---

def main():
    print("-----------------------------------------------------------------")
    print(" T-Shirt Color (via OpenAI API) to Hue Lights Sync (Room Mode) ")
    print("-----------------------------------------------------------------")
    print(f"This script uses your webcam and OpenAI API ({OPENAI_MODEL}) to detect your shirt color")
    print(f"and sets Philips Hue lights in the room '{TARGET_HUE_ROOM_NAME}' accordingly at MAX brightness.")
    print("\nPrerequisites:")
    print(" - opencv-python, numpy, phue, requests, math libraries installed.")
    print(" - Philips Hue Bridge on the same network.")
    print(" - OpenAI API Key (script will prompt if not found).")
    print(" - You might need to press the link button on your Hue bridge.")
    print(f" - Ensure TARGET_HUE_ROOM_NAME ('{TARGET_HUE_ROOM_NAME}') is correctly set in this script.")
    print("-----------------------------------------------------------------\n")

    effective_bridge_ip = HUE_BRIDGE_IP or get_config_value(USER_IP_CONFIG_FILE, "Enter your Philips Hue Bridge IP address (e.g., 192.168.1.100): ")
    if not effective_bridge_ip: print("Could not obtain Hue Bridge IP. Exiting."); return

    effective_openai_api_key = OPENAI_API_KEY
    if not effective_openai_api_key: 
        if os.path.exists(USER_OPENAI_KEY_CONFIG_FILE):
             with open(USER_OPENAI_KEY_CONFIG_FILE, 'r') as f: key_from_file = f.read().strip()
             if key_from_file: effective_openai_api_key = key_from_file; print(f"Found OpenAI API Key in '{USER_OPENAI_KEY_CONFIG_FILE}'.")
        if not effective_openai_api_key: 
            print("\n--- OpenAI API Key Setup ---")
            effective_openai_api_key = input("Enter your OpenAI API Key: ").strip()
            if effective_openai_api_key:
                with open(USER_OPENAI_KEY_CONFIG_FILE, 'w') as f: f.write(effective_openai_api_key)
                print(f"Saved OpenAI API Key to '{USER_OPENAI_KEY_CONFIG_FILE}'.")
            else: print("No OpenAI API Key provided. Exiting."); return
    if not effective_openai_api_key: print("OpenAI API Key is missing. Exiting."); return

    bridge = connect_to_hue_bridge(effective_bridge_ip)
    if not bridge: print("Failed to connect to Hue Bridge. Exiting."); return

    if bridge and TARGET_HUE_ROOM_NAME:
        print(f"\nChecking for target Hue Room: '{TARGET_HUE_ROOM_NAME}'...")
        try:
            target_room_id = bridge.get_group_id_by_name(TARGET_HUE_ROOM_NAME)
            if target_room_id is not None:
                group_info = bridge.get_group(target_room_id)
                lights_in_room = group_info.get('lights', [])
                print(f"Successfully found Hue Room: '{TARGET_HUE_ROOM_NAME}' (ID: {target_room_id}) with {len(lights_in_room)} light(s).")
            else:
                print(f"Warning: Target Hue Room '{TARGET_HUE_ROOM_NAME}' not found on the bridge.")
                print("Please check the TARGET_HUE_ROOM_NAME setting in the script and ensure it matches a room in your Philips Hue app.")
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
            print(f"Error when checking for Hue Room '{TARGET_HUE_ROOM_NAME}': {e}")
            print("Proceeding, but light changes may fail.")
    elif not TARGET_HUE_ROOM_NAME:
        print("\nWarning: TARGET_HUE_ROOM_NAME is not set in the script. Light changes will not be targeted to a specific room and will likely fail.")


    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: Could not open webcam."); return

    print("\nWebcam opened successfully.")
    print("Instructions:")
    print(" - Ensure your upper body clothing is clearly visible in the webcam feed.")
    print(f" - Press 'c' to capture color and set Hue lights in room '{TARGET_HUE_ROOM_NAME}' (can take a few seconds).")
    print(" - Press 'q' to quit.")
    print("-----------------------------------------------------------------\n")
    
    show_processing_message = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None: print("Error: Can't receive frame. Exiting ..."); break

        display_frame = frame.copy()
        display_frame = cv2.flip(display_frame, 1) 

        if show_processing_message:
            cv2.putText(display_frame, f"Processing with OpenAI ({OPENAI_MODEL})...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(f'Webcam - Shirt Color Detector (Room: {TARGET_HUE_ROOM_NAME})', display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('c'):
            print(f"\nCapturing color with OpenAI API ({OPENAI_MODEL})...")
            show_processing_message = True
            temp_display_frame = display_frame.copy()
            cv2.putText(temp_display_frame, f"Processing with OpenAI ({OPENAI_MODEL})...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(f'Webcam - Shirt Color Detector (Room: {TARGET_HUE_ROOM_NAME})', temp_display_frame)
            cv2.waitKey(1) 

            detected_rgb_color = get_dominant_color_from_openai(frame, effective_openai_api_key)
            show_processing_message = False

            if detected_rgb_color:
                set_hue_lights_color(bridge, detected_rgb_color, TARGET_HUE_ROOM_NAME)
            else:
                print("Could not determine color using OpenAI API. Try again or check API key/quota.")

        elif key == ord('q'):
            print("Quitting application..."); break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()

