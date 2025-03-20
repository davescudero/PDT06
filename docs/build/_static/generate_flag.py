#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para generar una imagen simple de la bandera de México para la documentación.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

def create_mexico_flag(save_path="mexico_flag.png", width=600, height=300):
    """
    Crea una imagen simple de la bandera de México y la guarda en la ruta especificada.
    
    Args:
        save_path (str): Ruta donde guardar la imagen
        width (int): Ancho de la imagen en píxeles
        height (int): Alto de la imagen en píxeles
    """
    # Configuración de la figura
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Crear las tres franjas verticales
    stripe_width = width / 3
    
    # Franja verde
    ax.add_patch(Rectangle((0, 0), stripe_width, height, facecolor='#006341', edgecolor=None))
    
    # Franja blanca
    ax.add_patch(Rectangle((stripe_width, 0), stripe_width, height, facecolor='white', edgecolor=None))
    
    # Franja roja
    ax.add_patch(Rectangle((2*stripe_width, 0), stripe_width, height, facecolor='#CE1126', edgecolor=None))
    
    # Añadir un círculo en el centro para simular el escudo
    center_x = width / 2
    center_y = height / 2
    circle_radius = min(height, width) * 0.15
    
    # Círculo marrón para simular el escudo
    ax.add_patch(Circle((center_x, center_y), circle_radius, facecolor='#754C24', edgecolor='black', linewidth=1))
    
    # Configuración del gráfico
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Guardar la imagen
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Bandera generada y guardada en {save_path}")

if __name__ == "__main__":
    create_mexico_flag("mexico_flag.png") 