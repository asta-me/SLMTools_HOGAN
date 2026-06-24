import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Ellipse
import tifffile  # Installabile via 'pip install tifffile'

def analyze_psf_2d(roi_data):
    """
    Esegue l'analisi bidimensionale dei momenti di una singola PSF.
    """
    img = roi_data.astype(float)
    
    # Sottrazione del background locale (stima basata sul minimo della ROI)
    # Fondamentale per evitare che il rumore di fondo pesi sul secondo momento
    bg_est = np.min(img)
    img_sub = img - bg_est
    
    H, W = img_sub.shape
    y, x = np.mgrid[0:H, 0:W]
    
    # Momento zero (Intensità totale integrata)
    m00 = np.sum(img_sub)
    if m00 == 0:
        return None
    
    # 1. Valore Massimo originale (non pre-sottratto) e posizione del picco
    max_val = np.max(roi_data)
    max_y, max_x = np.unravel_index(np.argmax(roi_data), roi_data.shape)
    
    # 2. Centro di Massa (Centroid)
    xc = np.sum(img_sub * x) / m00
    yc = np.sum(img_sub * y) / m00
    
    # 3. Momenti centrali del secondo ordine
    mu20 = np.sum(img_sub * (x - xc)**2) / m00
    mu02 = np.sum(img_sub * (y - yc)**2) / m00
    mu11 = np.sum(img_sub * (x - xc) * (y - yc)) / m00
    
    # 4. Secondo momento totale (varianza spaziale radiale)
    m2_radial = mu20 + mu02
    
    # 5. Analisi degli assi principali (Autovalori della matrice di covarianza)
    # Risoluzione analitica del polinomio caratteristico per una matrice 2x2
    trace = mu20 + mu02
    term = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
    
    lambda1 = (trace + term) / 2.0  # Varianza asse maggiore
    lambda2 = (trace - term) / 2.0  # Varianza asse minore
    
    # Sanity check numerico per fluttuazioni vicino a zero
    lambda1 = max(0.0, lambda1)
    lambda2 = max(0.0, lambda2)
    
    sigma_major = np.sqrt(lambda1)
    sigma_minor = np.sqrt(lambda2)
    
    # 6. Rotondità (Roundness) -> 1 significa perfettamente circolare
    roundness = sigma_minor / sigma_major if sigma_major > 0 else 1.0
    
    # 7. Angolo di orientazione dell'asse maggiore (in gradi)
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    theta_deg = np.degrees(theta)
    
    # 8. Stima FWHM (sotto approssimazione Gaussiana: FWHM = 2*sqrt(2*ln2)*sigma)
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))  # ~2.355
    fwhm_major = fwhm_factor * sigma_major
    fwhm_minor = fwhm_factor * sigma_minor
    
    return {
        'max_intensity': max_val,
        'peak_index': (max_x, max_y),
        'centroid': (xc, yc),
        'second_moment_radial': m2_radial,
        'sigma_major': sigma_major,
        'sigma_minor': sigma_minor,
        'roundness': roundness,
        'orientation_deg': theta_deg,
        'fwhm_major': fwhm_major,
        'fwhm_minor': fwhm_minor
    }

class PSFSelectorInteractive:
    def __init__(self, image_path, crop_half=7):
        # Carica l'immagine TIFF (supporta 8, 12, 16 bit nativamente)
        self.full_image = tifffile.imread(image_path)
        # Se l'immagine è RGB o multi-canale, converti in scala di grigi
        if self.full_image.ndim == 3:
            self.full_image = self.full_image.mean(axis=-1)
        # Semilato della finestra quadrata centrata sul picco per l'analisi (finestra = 2*crop_half+1)
        self.crop_half = crop_half
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(self.full_image, cmap='gray', origin='upper')
        self.ax.set_title("Trascina il mouse per selezionare la ROI della PSF")
        
        # Attiva il selettore rettangolare
        self.rs = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],  # Tasto sinistro del mouse
            minspanx=5, minspany=5,
            spancoords='pixels',
            props=dict(facecolor='none', edgecolor='cyan', linewidth=1.5, alpha=0.7)
        )
        self.ellipse_patch = None
        self.centroid_patch = None
        self.axis_lines = []
        plt.show()

    def on_select(self, eclick, erelease):
        # Estrai le coordinate della ROI selezionata dall'utente
        x1, y1 = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
        x2, y2 = int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))
        
        roi_data = self.full_image[y1:y2, x1:x2]
        if roi_data.size == 0:
            return
        
        # Trova il picco nella ROI e converti in coordinate globali
        peak_y_roi, peak_x_roi = np.unravel_index(np.argmax(roi_data), roi_data.shape)
        peak_x_global = x1 + peak_x_roi
        peak_y_global = y1 + peak_y_roi
        
        # Ritaglia una finestra quadrata (2*crop_half+1) centrata sul picco
        h = self.crop_half
        img_h, img_w = self.full_image.shape
        cy0 = max(0, peak_y_global - h)
        cy1 = min(img_h, peak_y_global + h + 1)
        cx0 = max(0, peak_x_global - h)
        cx1 = min(img_w, peak_x_global + h + 1)
        analysis_crop = self.full_image[cy0:cy1, cx0:cx1]
        
        print(f"\nPicco trovato a ({peak_x_global}, {peak_y_global}) — analisi su finestra {analysis_crop.shape[1]}x{analysis_crop.shape[0]} pixel")
        
        metrics = analyze_psf_2d(analysis_crop)
        if metrics is None:
            print("ROI non valida o intensità nulla.")
            return
        
        # Converte le coordinate locali del crop in coordinate globali
        global_xc = cx0 + metrics['centroid'][0]
        global_yc = cy0 + metrics['centroid'][1]
        
        # Stampa il Report Ottico a terminale
        print("\n" + "="*40)
        print("          REPORT OTTICO DELLA PSF          ")
        print("="*40)
        print(f"Massimo Assoluto (I_max):     {metrics['max_intensity']:.1f}")
        print(f"Centro di Massa (X, Y):       ({global_xc:.2f}, {global_yc:.2f}) pixel")
        print(f"Secondo Momento Radiale (M2): {metrics['second_moment_radial']:.3f} pixel²")
        print(f"Deviazione Standard Asse Magg: {metrics['sigma_major']:.3f} pixel")
        print(f"Deviazione Standard Asse Min:  {metrics['sigma_minor']:.3f} pixel")
        print(f"Rotondità (Roundness):         {metrics['roundness']:.3f} (1.0 = cerchio perfetto)")
        print(f"Orientazione Asse Maggiore:   {metrics['orientation_deg']:.1f}°")
        print(f"FWHM Stimata Asse Maggiore:   {metrics['fwhm_major']:.2f} pixel")
        print(f"FWHM Stimata Asse Minore:     {metrics['fwhm_minor']:.2f} pixel")
        print("="*40)
        
        # Aggiorna gli overlay grafici sull'immagine
        if self.ellipse_patch:
            self.ellipse_patch.remove()
        if self.centroid_patch:
            self.centroid_patch.remove()
        for line in self.axis_lines:
            line.remove()
        self.axis_lines = []
            
        # Disegna l'ellisse basata su 2*sigma (contiene ~86% dell'energia per una Gaussiana 2D)
        self.ellipse_patch = Ellipse(
            xy=(global_xc, global_yc),
            width=4 * metrics['sigma_major'],
            height=4 * metrics['sigma_minor'],
            angle=metrics['orientation_deg'],
            edgecolor='red', facecolor='none', linewidth=2, linestyle='--',
            label='Profilo 2-Sigma'
        )
        self.centroid_patch = self.ax.plot(global_xc, global_yc, 'rx', markersize=8)[0]
        self.ax.add_patch(self.ellipse_patch)

        # Disegna i segmenti degli assi maggiore (giallo) e minore (ciano)
        theta_rad = np.radians(metrics['orientation_deg'])
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        dx_maj = 2 * metrics['sigma_major'] * cos_t
        dy_maj = 2 * metrics['sigma_major'] * sin_t
        l1, = self.ax.plot(
            [global_xc - dx_maj, global_xc + dx_maj],
            [global_yc - dy_maj, global_yc + dy_maj],
            color='yellow', linewidth=1.5
        )
        dx_min = 2 * metrics['sigma_minor'] * (-sin_t)
        dy_min = 2 * metrics['sigma_minor'] * cos_t
        l2, = self.ax.plot(
            [global_xc - dx_min, global_xc + dx_min],
            [global_yc - dy_min, global_yc + dy_min],
            color='cyan', linewidth=1.5
        )
        self.axis_lines = [l1, l2]
        self.fig.canvas.draw_idle()

        # Zoom 30x30 centrato sul centroide in una nuova figura
        self._show_zoom(global_xc, global_yc, metrics, zoom_half=15)

    def _show_zoom(self, global_xc, global_yc, metrics, zoom_half=15):
        img_h, img_w = self.full_image.shape
        zx0 = max(0, int(round(global_xc)) - zoom_half)
        zx1 = min(img_w, int(round(global_xc)) + zoom_half)
        zy0 = max(0, int(round(global_yc)) - zoom_half)
        zy1 = min(img_h, int(round(global_yc)) + zoom_half)
        zoom_data = self.full_image[zy0:zy1, zx0:zx1]

        fig_z, ax_z = plt.subplots(figsize=(5, 5))
        ax_z.imshow(zoom_data, cmap='gray', origin='upper',
                    extent=[zx0, zx1, zy1, zy0])
        ax_z.set_title(f"Zoom PSF — centroide ({global_xc:.1f}, {global_yc:.1f})")

        # Ellisse 2-sigma
        ellipse_z = Ellipse(
            xy=(global_xc, global_yc),
            width=4 * metrics['sigma_major'],
            height=4 * metrics['sigma_minor'],
            angle=metrics['orientation_deg'],
            edgecolor='red', facecolor='none', linewidth=1.5, linestyle='--'
        )
        ax_z.add_patch(ellipse_z)

        # Asse maggiore (giallo) e minore (ciano)
        theta_rad = np.radians(metrics['orientation_deg'])
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        dx_maj = 2 * metrics['sigma_major'] * cos_t
        dy_maj = 2 * metrics['sigma_major'] * sin_t
        ax_z.plot([global_xc - dx_maj, global_xc + dx_maj],
                  [global_yc - dy_maj, global_yc + dy_maj],
                  color='yellow', linewidth=1.5)
        dx_min = 2 * metrics['sigma_minor'] * (-sin_t)
        dy_min = 2 * metrics['sigma_minor'] * cos_t
        ax_z.plot([global_xc - dx_min, global_xc + dx_min],
                  [global_yc - dy_min, global_yc + dy_min],
                  color='cyan', linewidth=1.5)

        # Centroide
        ax_z.plot(global_xc, global_yc, 'rx', markersize=10, markeredgewidth=1.5)

        ax_z.set_xlim(zx0, zx1)
        ax_z.set_ylim(zy1, zy0)
        fig_z.tight_layout()
        plt.show()

# Esempio di utilizzo (sostituisci col percorso del tuo file TIFF)
selector = PSFSelectorInteractive(r"C:\Users\astam\Desktop\OneDrive - Politecnico di Milano\Polimi\Datasets\Phase_Retrieval\26_06_11_Results\PC_Corrected.tiff", crop_half=20)