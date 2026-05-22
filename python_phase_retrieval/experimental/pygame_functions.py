# -*- coding: utf-8 -*-
"""Local pygame utilities for SLM display in experimental scripts."""

from __future__ import annotations

import os

import numpy as np
import pygame


def init_pygame(screen_index: int = 0):
    from screeninfo import get_monitors

    if get_monitors()[screen_index].is_primary:
        print("Warning, selected screen is primary")
    pygame.init()
    width, height = pygame.display.get_desktop_sizes()[screen_index]
    window = pygame.display.set_mode((width, height), pygame.NOFRAME, display=screen_index)
    return window


def close_pygame() -> None:
    pygame.quit()


def display_numpy_hologram(hologram: np.ndarray, window) -> None:
    slm_size = (window.get_size()[1], window.get_size()[0])
    if hologram.shape[0] > slm_size[0] or hologram.shape[1] > slm_size[1]:
        print("The hologram is too big")

    array = np.stack((hologram, hologram, hologram), axis=2)
    surf = pygame.surfarray.make_surface(array)
    surf_rect = surf.get_rect()
    surf_rect.center = window.get_rect().center
    window.blit(surf, surf_rect)
    pygame.display.flip()


def display_bmp_hologram(filename: str, window) -> None:
    image = pygame.image.load(filename)
    window_width, window_height = window.get_size()
    image_width, image_height = image.get_size()
    x = (window_width - image_width) // 2
    y = (window_height - image_height) // 2
    window.blit(image, (x, y))
    pygame.display.flip()


def display_bmp_video(folder_path: str, framerate: float, window) -> None:
    bmp_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".bmp")]
    bmp_files.sort()
    clock = pygame.time.Clock()

    for i, bmp_file in enumerate(bmp_files):
        image = pygame.image.load(os.path.join(folder_path, bmp_file))
        if i == 0:
            window.fill((0, 0, 0))
            window_width, window_height = window.get_size()
            image_width, image_height = image.get_size()
            x = (window_width - image_width) // 2
            y = (window_height - image_height) // 2
        window.blit(image, (x, y))
        pygame.display.flip()
        clock.tick(framerate)


def navigate_bmp_frames(folder_path: str, window) -> None:
    bmp_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".bmp")]
    bmp_files.sort()
    if not bmp_files:
        raise ValueError(f"No BMP files found in {folder_path}")

    current_frame = 0
    image = pygame.image.load(os.path.join(folder_path, bmp_files[current_frame]))
    window_width, window_height = window.get_size()
    image_width, image_height = image.get_size()
    x = (window_width - image_width) // 2
    y = (window_height - image_height) // 2
    window.blit(image, (x, y))
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_frame = (current_frame + 1) % len(bmp_files)
                elif event.key == pygame.K_LEFT:
                    current_frame = (current_frame - 1) % len(bmp_files)
                elif event.key == pygame.K_ESCAPE:
                    running = False

                image = pygame.image.load(os.path.join(folder_path, bmp_files[current_frame]))
                window.blit(image, (x, y))
                pygame.display.flip()


def display_bmp_video_loop(folder_path: str, framerate: float, window) -> None:
    while True:
        display_bmp_video(folder_path, framerate, window)
