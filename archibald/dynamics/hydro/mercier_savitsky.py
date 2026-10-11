"""
Mercier & Savitsky (1973), "Resistance of transom-stern craft in the pre-planing
regime", Davidson Lab Report 1667 / DTIC AD-764 958.

Régression sur 118 carènes (7 séries) :  R/W = f(Fn_vol, L/∇^(1/3), W/(ρ g b³)
[charge par largeur], angle d'entrée, Atr/Ax).  Domaine : Fn_vol < 2 (usage
1-2), demi-angle d'entrée 10-55°, L/∇^(1/3) 2-12.

!! LES COEFFICIENTS NE SONT PAS INCLUS : le rapport n'a pas pu être consulté
   (accès bloqué).  Renseigner MERCIER_SAVITSKY_COEFFS puis implémenter
   _regression().  Tant que la table est vide, un NotImplementedError explicite
   est levé (le registre n'expose alors pas cette méthode par défaut).
"""
import archibald.numpy as np
from planing_common import speed_ms, Fn_vol

MERCIER_SAVITSKY_COEFFS = None     # à renseigner depuis le rapport original
AVAILABLE = False


def _regression(Fn, slender, beam_loading, ie_deg, Atr_Ax):
    raise NotImplementedError(
        "Coefficients de Mercier & Savitsky (1973) non renseignés : voir le "
        "docstring du module (DTIC AD-764 958).")


def compute_R_over_W_mercier_savitsky(stw, g, volume, Lwl, Bwl, ie, Atr, Cx, Aws=None, **kw):
    """R/W total ; à brancher sur _regression() une fois la table disponible."""
    V = speed_ms(stw)
    slender = Lwl / volume ** (1. / 3.)
    return _regression(Fn_vol(V, volume, g), slender, kw.get("beam_loading"),
                       ie, Atr / (Cx * Bwl * kw.get("T", 1.)))
