# Contains classes and methods for allowing control with a gamepad.

from pynput import keyboard
import time
import numpy as np
import transforms3d as tf3d

keyboard_event_dict = {
    'w': 'w',
    'a': 'a',
    's': 's',
    'd': 'd',
    'e': 'e',
    'q': 'q',
    'u': 'u',
    'j': 'j',
    'i': 'i',
    'k': 'k',
    'o': 'o',
    'l': 'l',
    'b': 'b',
    'g': 'g',
    'f': 'f',
    't': 't',
    'space': 'space',
    'r': 'r',
    'enter': 'enter',
    'backspace': 'backspace',
    'right shift': 'r_shift',
}

class KeyboardSteer:
    """Class for steering the robot with a keyboard."""
    STICKS = ['LX', 'RX', 'LY', 'RY']
    TRIGS = ['LT', 'RT']

    def __init__(self, trans_rotation=None, action_multiplier=.3):
        self.btn_state = {value: False for value in keyboard_event_dict.values()}
        self.old_btn_state = self.btn_state.copy()
        self.action_multiplier = action_multiplier
        self.trans_rotation = np.eye(4) if trans_rotation is None else tf3d.euler.euler2mat(*trans_rotation, axes='sxyz')

        self.gripper_toggle = False
        self.enter_toggle = False
        self.space_toggle = False
        self.d_pressed = False
        self.s_toggle = False
        self.right_start_time = None

        # Setup the keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            key_str = key.char
        except AttributeError:
            key_str = key.name

        if key_str in keyboard_event_dict:
            self.btn_state[keyboard_event_dict[key_str]] = True

    def _on_release(self, key):
        try:
            key_str = key.char
        except AttributeError:
            key_str = key.name

        if key_str in keyboard_event_dict:
            self.btn_state[keyboard_event_dict[key_str]] = False

    def process_events(self):
        """Process available events. Call this one."""
        self._set_old_button_states()
        # No need to manually read events with pynput, it handles events via callbacks

    def _set_old_button_states(self):
        self.old_btn_state = self.btn_state.copy()

    def normalize_state(self):
        """Make sticks be between -1 and 1, and trigs be between 0 and 1."""
        # This is not used in the current example, but you can implement it if needed
        pass

    def move_robot(self):
        trans_vel = self.action_multiplier * np.array([
            self.btn_state['d'] - self.btn_state['a'],
            self.btn_state['w'] - self.btn_state['s'],
            self.btn_state['e'] - self.btn_state['q']
        ])
        rot_vel = self.action_multiplier * np.array([
            self.btn_state['u'] - self.btn_state['j'],
            self.btn_state['i'] - self.btn_state['k'],
            self.btn_state['o'] - self.btn_state['l']
        ])
        grip = self.btn_state['space']

        return trans_vel, rot_vel, grip

    def _handle_button_toggles(self):
        """ Handle button toggles. Not a long-term solution. """
        cur_time = time.time()
        hold_time = 2

        if self.btn_state['enter'] and not self.old_btn_state['enter']:
            self.enter_toggle = not self.enter_toggle

        if self.btn_state['space'] and not self.old_btn_state['space']:
            self.space_toggle = True

        if self.btn_state['d'] and not self.old_btn_state['d']:
            self.d_pressed = True

        if self.btn_state['r_shift'] and not self.old_btn_state['r_shift']:
            self.gripper_toggle = not self.gripper_toggle

        if self.btn_state['s'] and not self.old_btn_state['s']:
            self.s_toggle = not self.s_toggle

        if self.btn_state['enter'] and not self.old_btn_state['enter']:
            self.right_start_time = cur_time

        if self.btn_state['enter'] and (cur_time - self.right_start_time > 2):
            self.enter_hold = True
        else:
            self.enter_hold = False


if __name__ == '__main__':
    ks = KeyboardSteer()

    try:
        while True:
            ks.process_events()
            # print(gs.normalized_btn_state['LX'], gs.btn_state['A'])
            print(ks.btn_state)
            print(bool(ks.btn_state['w']))
            time.sleep(.1)
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")

