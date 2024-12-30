#!/bin/bash
ssh -t -t flyprojection@flyprojection-server "/bin/sh -c 'cd ws2812b_controller/;sudo python3 LEDClient.py;'"
