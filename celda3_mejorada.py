"""
CELDA 3 MEJORADA - Bounding Box con separación antena/soporte

Mejoras implementadas:
1. Comparación de Trimesh vs Open3D para cálculo de OBB
2. Detección de plano de mejor ajuste a la superficie frontal
3. Análisis de densidad con ventanas deslizantes de 5cm
4. Detección de cambios bruscos (derivada) para separar antena del soporte
5. Filtrado de puntos del soporte antes de calcular bbox final

Autor: Claude
Fecha: 2025-11-14
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from sklearn.linear_model import RANSACRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SELECCIÓN DE FAMILIA
# ============================================================================

print("\n" + "=" * 60)
print("ANÁLISIS DETALLADO POR FAMILIA - VERSIÓN MEJORADA")
print("=" * 60)

familias_disponibles = sorted(df_antenas['Familia'].unique())
print("\nFamilias disponibles:")
for i, familia in enumerate(familias_disponibles):
    n_ejemplares = len(df_antenas[df_antenas['Familia'] == familia]['Ejemplar'].unique())
    n_puntos = len(df_antenas[df_antenas['Familia'] == familia])
    print(f"  [{i}] Familia {int(familia)}: {n_ejemplares} ejemplares, {n_puntos} puntos")

familia_idx = int(input("\n¿Qué familia deseas procesar? [índice]: "))
familia_seleccionada = familias_disponibles[familia_idx]

print(f"\n✓ Familia {int(familia_seleccionada)} seleccionada")

# Filtrar datos de la familia seleccionada
df_familia = df_antenas[df_antenas['Familia'] == familia_seleccionada].copy()
ejemplares = sorted(df_familia['Ejemplar'].unique())

print(f"✓ Ejemplares en esta familia: {[int(e) for e in ejemplares]}")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def limpiar_outliers(puntos, std_threshold=2.5):
    """Elimina outliers usando desviación estándar"""
    if len(puntos) < 4:
        return puntos

    centro = np.mean(puntos, axis=0)
    distancias = np.linalg.norm(puntos - centro, axis=1)

    media_dist = np.mean(distancias)
    std_dist = np.std(distancias)

    mask = distancias < (media_dist + std_threshold * std_dist)
    return puntos[mask]


def ajustar_plano_ransac(puntos, residual_threshold=0.05, max_trials=1000):
    """
    Ajusta un plano a los puntos usando RANSAC para robustez contra outliers.

    El plano se define como: ax + by + cz + d = 0
    donde (a, b, c) es el vector normal (normalizado)

    Args:
        puntos: array Nx3 de puntos
        residual_threshold: distancia máxima para considerar un punto como inlier (metros)
        max_trials: número máximo de iteraciones RANSAC

    Returns:
        dict con:
        - normal: vector normal al plano (normalizado)
        - punto_en_plano: un punto que pertenece al plano
        - d: coeficiente d de la ecuación del plano
        - inliers_mask: máscara booleana de inliers
        - n_inliers: número de inliers
    """
    if len(puntos) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para ajustar un plano")

    # Preparar datos para RANSAC
    X = puntos[:, :2]  # X, Y
    y = puntos[:, 2]   # Z

    # RANSAC para ajustar Z = aX + bY + c
    ransac = RANSACRegressor(
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=42
    )

    ransac.fit(X, y)

    # Extraer coeficientes
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    # El plano es: Z = aX + bY + c
    # Reescribir como: aX + bY - Z + c = 0
    # Vector normal (sin normalizar): (a, b, -1)
    normal_sin_normalizar = np.array([a, b, -1])
    normal = normal_sin_normalizar / np.linalg.norm(normal_sin_normalizar)

    # Coeficiente d de la ecuación ax + by + cz + d = 0
    # Como tenemos aX + bY - Z + c = 0, entonces d = c
    d = c

    # Punto en el plano (usar el centroide de los inliers)
    inliers_mask = ransac.inlier_mask_
    puntos_inliers = puntos[inliers_mask]
    punto_en_plano = np.mean(puntos_inliers, axis=0)

    n_inliers = np.sum(inliers_mask)
    porcentaje_inliers = 100 * n_inliers / len(puntos)

    print(f"    Plano ajustado: {n_inliers}/{len(puntos)} inliers ({porcentaje_inliers:.1f}%)")
    print(f"    Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")

    return {
        'normal': normal,
        'punto_en_plano': punto_en_plano,
        'd': d,
        'inliers_mask': inliers_mask,
        'n_inliers': n_inliers
    }


def calcular_distancias_al_plano(puntos, plano_info):
    """
    Calcula la distancia con signo de cada punto al plano.

    Distancia positiva: lado de la normal
    Distancia negativa: lado opuesto a la normal

    Args:
        puntos: array Nx3
        plano_info: dict con 'normal' y 'punto_en_plano'

    Returns:
        array de distancias con signo
    """
    normal = plano_info['normal']
    punto_plano = plano_info['punto_en_plano']

    # Distancia con signo = (punto - punto_plano) · normal
    vectores = puntos - punto_plano
    distancias = np.dot(vectores, normal)

    return distancias


def analizar_densidad_perpendicular(puntos, plano_info, ancho_ventana=0.05,
                                     max_distancia=0.5):
    """
    Analiza la densidad de puntos en ventanas perpendiculares al plano.

    Args:
        puntos: array Nx3
        plano_info: información del plano de referencia
        ancho_ventana: ancho de cada ventana en metros (default 5cm)
        max_distancia: distancia máxima a analizar en cada dirección (metros)

    Returns:
        dict con:
        - distancias: centros de las ventanas (con signo)
        - densidades: densidad (puntos/m³) en cada ventana
        - cuentas: número de puntos en cada ventana
        - volumenes: volumen de cada ventana
    """
    # Calcular distancias de todos los puntos al plano
    distancias_puntos = calcular_distancias_al_plano(puntos, plano_info)

    # Crear bins para las ventanas
    bins = np.arange(-max_distancia, max_distancia + ancho_ventana, ancho_ventana)
    centros_ventanas = (bins[:-1] + bins[1:]) / 2

    # Contar puntos en cada ventana
    cuentas, _ = np.histogram(distancias_puntos, bins=bins)

    # Estimar área de la sección transversal (usar convex hull en proyección 2D)
    # Proyectar puntos al plano
    normal = plano_info['normal']

    # Crear sistema de coordenadas del plano
    # Eje 1: perpendicular a normal y a Z
    if abs(normal[2]) < 0.9:
        eje1 = np.cross(normal, [0, 0, 1])
    else:
        eje1 = np.cross(normal, [1, 0, 0])
    eje1 = eje1 / np.linalg.norm(eje1)

    # Eje 2: perpendicular a normal y eje1
    eje2 = np.cross(normal, eje1)
    eje2 = eje2 / np.linalg.norm(eje2)

    # Proyectar puntos al plano
    punto_plano = plano_info['punto_en_plano']
    vectores = puntos - punto_plano
    coords_plano = np.column_stack([
        np.dot(vectores, eje1),
        np.dot(vectores, eje2)
    ])

    # Calcular convex hull para estimar área
    try:
        hull = ConvexHull(coords_plano)
        area_transversal = hull.volume  # En 2D, 'volume' es el área
    except:
        # Si falla, usar bounding box como aproximación
        min_coords = np.min(coords_plano, axis=0)
        max_coords = np.max(coords_plano, axis=0)
        area_transversal = np.prod(max_coords - min_coords)

    # Calcular volúmenes y densidades
    volumenes = area_transversal * ancho_ventana
    densidades = cuentas / volumenes if volumenes > 0 else np.zeros_like(cuentas)

    print(f"    Análisis de densidad:")
    print(f"    - Área transversal estimada: {area_transversal:.4f} m²")
    print(f"    - Ancho ventana: {ancho_ventana*100:.1f} cm")
    print(f"    - Número de ventanas: {len(centros_ventanas)}")
    print(f"    - Rango analizado: [{-max_distancia:.2f}, {max_distancia:.2f}] m")

    return {
        'distancias': centros_ventanas,
        'densidades': densidades,
        'cuentas': cuentas,
        'volumenes': volumenes,
        'area_transversal': area_transversal
    }


def detectar_cambio_brusco_densidad(densidad_info, umbral_derivada=None,
                                     lado='positivo'):
    """
    Detecta el cambio brusco en la densidad usando derivadas.

    Args:
        densidad_info: resultado de analizar_densidad_perpendicular
        umbral_derivada: umbral para detectar cambio brusco (auto si None)
        lado: 'positivo' o 'negativo' (lado del plano a analizar)

    Returns:
        dict con:
        - distancia_corte: distancia al plano donde se detecta el cambio
        - indice_corte: índice en el array de distancias
        - derivada_max: valor máximo de la derivada (en valor absoluto)
    """
    distancias = densidad_info['distancias']
    densidades = densidad_info['densidades']

    # Filtrar por lado
    if lado == 'positivo':
        mask = distancias >= 0
    else:
        mask = distancias < 0

    dist_lado = distancias[mask]
    dens_lado = densidades[mask]

    if len(dens_lado) < 3:
        print(f"    ⚠️ Insuficientes puntos en lado {lado}")
        return None

    # Calcular derivada (diferencias finitas)
    derivada = np.diff(dens_lado) / np.diff(dist_lado)

    # Suavizar derivada (media móvil simple, ventana de 3)
    if len(derivada) >= 3:
        derivada_suavizada = np.convolve(derivada, np.ones(3)/3, mode='valid')
        dist_derivada = dist_lado[1:-1]  # Ajustar índices por el suavizado
    else:
        derivada_suavizada = derivada
        dist_derivada = dist_lado[1:]

    # Detectar cambio brusco
    # Buscamos la mayor caída en densidad (derivada más negativa)
    if len(derivada_suavizada) == 0:
        print(f"    ⚠️ No se pudo calcular derivada en lado {lado}")
        return None

    idx_min_derivada = np.argmin(derivada_suavizada)
    derivada_min = derivada_suavizada[idx_min_derivada]

    # Umbral automático si no se especifica
    if umbral_derivada is None:
        # Usar percentil 10 de las derivadas negativas
        derivadas_negativas = derivada_suavizada[derivada_suavizada < 0]
        if len(derivadas_negativas) > 0:
            umbral_derivada = np.percentile(derivadas_negativas, 10)
        else:
            umbral_derivada = -1e-6  # Valor por defecto muy pequeño

    # Verificar si hay un cambio significativo
    if derivada_min < umbral_derivada:
        distancia_corte = dist_derivada[idx_min_derivada]
        print(f"    ✓ Cambio brusco detectado en lado {lado}:")
        print(f"      - Distancia al plano: {distancia_corte:.3f} m")
        print(f"      - Derivada mínima: {derivada_min:.2f} puntos/m⁴")

        return {
            'distancia_corte': distancia_corte,
            'indice_corte': idx_min_derivada,
            'derivada_min': derivada_min,
            'derivada_completa': derivada_suavizada,
            'distancias_derivada': dist_derivada
        }
    else:
        print(f"    ⚠️ No se detectó cambio brusco significativo en lado {lado}")
        print(f"      - Derivada mínima: {derivada_min:.2f}")
        print(f"      - Umbral: {umbral_derivada:.2f}")
        return None


def separar_antena_soporte(puntos, plano_info, densidad_info,
                           lado_antena='positivo', metodo='auto'):
    """
    Separa los puntos de la antena de los del soporte usando análisis de densidad.

    Args:
        puntos: array Nx3 de puntos
        plano_info: información del plano de la superficie frontal
        densidad_info: resultado del análisis de densidad
        lado_antena: 'positivo' o 'negativo' (lado donde está la antena)
        metodo: 'auto', 'derivada', o 'umbral_fijo'

    Returns:
        dict con:
        - puntos_antena: puntos de la antena solamente
        - puntos_soporte: puntos del soporte
        - distancia_corte: distancia al plano usada para el corte
        - mask_antena: máscara booleana de puntos de antena
    """
    # Calcular distancias de todos los puntos al plano
    distancias = calcular_distancias_al_plano(puntos, plano_info)

    if metodo == 'auto' or metodo == 'derivada':
        # Detectar cambio brusco
        cambio_info = detectar_cambio_brusco_densidad(
            densidad_info,
            lado=lado_antena
        )

        if cambio_info is not None:
            distancia_corte = cambio_info['distancia_corte']
        else:
            # Fallback: usar un percentil conservador
            if lado_antena == 'positivo':
                distancia_corte = np.percentile(distancias[distancias > 0], 75)
            else:
                distancia_corte = np.percentile(distancias[distancias < 0], 25)
            print(f"    ℹ️ Usando corte por percentil: {distancia_corte:.3f} m")

    elif metodo == 'umbral_fijo':
        # Usar umbral fijo (ej: 0.15m desde el plano)
        distancia_corte = 0.15 if lado_antena == 'positivo' else -0.15
        print(f"    ℹ️ Usando umbral fijo: {distancia_corte:.3f} m")

    else:
        raise ValueError(f"Método '{metodo}' no reconocido")

    # Aplicar corte
    if lado_antena == 'positivo':
        # Antena en lado positivo: mantener puntos con distancia >= 0 y <= distancia_corte
        mask_antena = (distancias >= 0) & (distancias <= abs(distancia_corte))
    else:
        # Antena en lado negativo: mantener puntos con distancia <= 0 y >= distancia_corte
        mask_antena = (distancias <= 0) & (distancias >= -abs(distancia_corte))

    puntos_antena = puntos[mask_antena]
    puntos_soporte = puntos[~mask_antena]

    print(f"    Separación antena/soporte:")
    print(f"    - Puntos antena: {len(puntos_antena)} ({100*len(puntos_antena)/len(puntos):.1f}%)")
    print(f"    - Puntos soporte: {len(puntos_soporte)} ({100*len(puntos_soporte)/len(puntos):.1f}%)")

    return {
        'puntos_antena': puntos_antena,
        'puntos_soporte': puntos_soporte,
        'distancia_corte': distancia_corte,
        'mask_antena': mask_antena
    }


def calcular_bounding_box_trimesh(puntos):
    """
    Calcula el Oriented Bounding Box (OBB) usando trimesh.

    Retorna dict con centro, dimensiones, ejes, etc.
    """
    import trimesh

    point_cloud = trimesh.PointCloud(puntos)
    to_origin, extents = trimesh.bounds.oriented_bounds(point_cloud)

    from_origin = np.linalg.inv(to_origin)
    centro = from_origin[:3, 3]
    ejes = from_origin[:3, :3]
    dimensiones = extents

    min_coords = -extents / 2
    max_coords = extents / 2
    volumen = np.prod(dimensiones)

    # Verificación
    puntos_homogeneos = np.column_stack([puntos, np.ones(len(puntos))])
    puntos_locales = (to_origin @ puntos_homogeneos.T).T[:, :3]
    dentro = np.all((puntos_locales >= min_coords - 0.001) &
                    (puntos_locales <= max_coords + 0.001), axis=1)
    puntos_fuera = np.sum(~dentro)

    return {
        'centro': centro,
        'dimensiones': dimensiones,
        'ejes': ejes,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'volumen': volumen,
        'puntos_fuera': puntos_fuera,
        'metodo': 'trimesh'
    }


def calcular_bounding_box_open3d(puntos, metodo='minimal'):
    """
    Calcula el Oriented Bounding Box usando Open3D.

    Args:
        puntos: array Nx3
        metodo: 'pca' o 'minimal'
            - 'pca': create_from_points() - basado en PCA del convex hull
            - 'minimal': create_from_points_minimal() - volumen mínimo (más lento)

    Retorna dict compatible con el formato de trimesh.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("    ⚠️ Open3D no está instalado. Instalar con: pip install open3d")
        return None

    # Crear PointCloud de Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(puntos)

    # Calcular OBB según método
    if metodo == 'minimal':
        obb = pcd.get_minimal_oriented_bounding_box()
    else:  # pca
        obb = pcd.get_oriented_bounding_box()

    # Extraer propiedades
    centro = np.asarray(obb.center)
    R = np.asarray(obb.R)  # Matriz de rotación 3x3
    extent = np.asarray(obb.extent)  # Dimensiones [x, y, z]

    # Calcular min/max en sistema local
    min_coords = -extent / 2
    max_coords = extent / 2
    volumen = np.prod(extent)

    # Verificar puntos dentro
    # Transformar puntos al sistema local del OBB
    puntos_centrados = puntos - centro
    puntos_locales = puntos_centrados @ R  # R es la matriz que rota al sistema local

    dentro = np.all((puntos_locales >= min_coords - 0.001) &
                    (puntos_locales <= max_coords + 0.001), axis=1)
    puntos_fuera = np.sum(~dentro)

    return {
        'centro': centro,
        'dimensiones': extent,
        'ejes': R,  # En Open3D, R ya son las columnas de los ejes
        'min_coords': min_coords,
        'max_coords': max_coords,
        'volumen': volumen,
        'puntos_fuera': puntos_fuera,
        'metodo': f'open3d_{metodo}'
    }


def calcular_bounding_box(puntos, metodo='trimesh', open3d_variant='minimal'):
    """
    Calcula el OBB usando el método especificado.

    Args:
        puntos: array Nx3
        metodo: 'trimesh', 'open3d_pca', 'open3d_minimal', o 'comparar'
        open3d_variant: 'pca' o 'minimal' (solo si metodo='open3d*')

    Returns:
        dict o lista de dicts (si metodo='comparar')
    """
    if metodo == 'trimesh':
        bb = calcular_bounding_box_trimesh(puntos)
        print(f"    Trimesh OBB: volumen = {bb['volumen']:.6f} m³, "
              f"puntos fuera = {bb['puntos_fuera']}")
        return bb

    elif metodo.startswith('open3d'):
        variant = metodo.split('_')[1] if '_' in metodo else open3d_variant
        bb = calcular_bounding_box_open3d(puntos, metodo=variant)
        if bb:
            print(f"    Open3D OBB ({variant}): volumen = {bb['volumen']:.6f} m³, "
                  f"puntos fuera = {bb['puntos_fuera']}")
        return bb

    elif metodo == 'comparar':
        print("    Comparando métodos de OBB:")
        resultados = []

        # Trimesh
        print("\n    [1] Trimesh:")
        bb_trimesh = calcular_bounding_box_trimesh(puntos)
        resultados.append(bb_trimesh)

        # Open3D PCA
        print("\n    [2] Open3D (PCA):")
        bb_o3d_pca = calcular_bounding_box_open3d(puntos, metodo='pca')
        if bb_o3d_pca:
            resultados.append(bb_o3d_pca)

        # Open3D Minimal
        print("\n    [3] Open3D (Minimal):")
        bb_o3d_min = calcular_bounding_box_open3d(puntos, metodo='minimal')
        if bb_o3d_min:
            resultados.append(bb_o3d_min)

        # Mostrar comparación
        print("\n    Comparación de volúmenes:")
        for i, bb in enumerate(resultados):
            print(f"      {i+1}. {bb['metodo']:20s}: {bb['volumen']:.6f} m³ "
                  f"(fuera: {bb['puntos_fuera']})")

        # Seleccionar el de menor volumen sin puntos fuera
        bb_validos = [bb for bb in resultados if bb['puntos_fuera'] == 0]
        if bb_validos:
            mejor_bb = min(bb_validos, key=lambda x: x['volumen'])
            print(f"\n    ✓ Mejor método: {mejor_bb['metodo']} (volumen mínimo sin puntos fuera)")
            return mejor_bb
        else:
            # Si todos tienen puntos fuera, elegir el de menos puntos fuera
            mejor_bb = min(resultados, key=lambda x: (x['puntos_fuera'], x['volumen']))
            print(f"\n    ⚠️ Todos los métodos tienen puntos fuera. "
                  f"Usando: {mejor_bb['metodo']}")
            return mejor_bb

    else:
        raise ValueError(f"Método '{metodo}' no reconocido. "
                        f"Use: 'trimesh', 'open3d_pca', 'open3d_minimal', o 'comparar'")


def obtener_vertices_box(bb_info):
    """
    Obtiene los 8 vértices del bounding box orientado.
    Compatible con formato de trimesh y open3d.
    """
    d = bb_info['dimensiones']
    min_c = bb_info['min_coords']

    # 8 vértices en coordenadas locales
    vertices_locales = np.array([
        # Base inferior (Z mínima)
        [min_c[0], min_c[1], min_c[2]],
        [min_c[0] + d[0], min_c[1], min_c[2]],
        [min_c[0] + d[0], min_c[1] + d[1], min_c[2]],
        [min_c[0], min_c[1] + d[1], min_c[2]],
        # Techo superior (Z máxima)
        [min_c[0], min_c[1], min_c[2] + d[2]],
        [min_c[0] + d[0], min_c[1], min_c[2] + d[2]],
        [min_c[0] + d[0], min_c[1] + d[1], min_c[2] + d[2]],
        [min_c[0], min_c[1] + d[1], min_c[2] + d[2]]
    ])

    # Transformar a coordenadas globales
    vertices_globales = bb_info['centro'] + vertices_locales @ bb_info['ejes'].T

    return vertices_globales


def identificar_cara_exterior(bb_info, puntos_limpios):
    """
    Identifica la cara del bounding box más alejada del eje vertical.
    (Código original de celda 3, sin modificaciones)
    """
    centro_box = bb_info['centro']
    ejes = bb_info['ejes']
    dims = bb_info['dimensiones']

    z_centro = centro_box[2]
    punto_eje_ref = np.array([0.0, 0.0, z_centro])

    caras_info = []
    nombres_caras = ['-X', '+X', '-Y', '+Y', '-Z', '+Z']

    for nombre in nombres_caras:
        vertices_cara = obtener_vertices_cara_por_nombre(nombre, bb_info)
        centro_cara = np.mean(vertices_cara, axis=0)

        if nombre == '-X':
            normal_local = np.array([-1, 0, 0])
            area = dims[1] * dims[2]
        elif nombre == '+X':
            normal_local = np.array([1, 0, 0])
            area = dims[1] * dims[2]
        elif nombre == '-Y':
            normal_local = np.array([0, -1, 0])
            area = dims[0] * dims[2]
        elif nombre == '+Y':
            normal_local = np.array([0, 1, 0])
            area = dims[0] * dims[2]
        elif nombre == '-Z':
            normal_local = np.array([0, 0, -1])
            area = dims[0] * dims[1]
        else:  # +Z
            normal_local = np.array([0, 0, 1])
            area = dims[0] * dims[1]

        normal_global = normal_local @ ejes.T
        distancia_al_punto_eje = np.linalg.norm(centro_cara - punto_eje_ref)

        caras_info.append({
            'nombre': nombre,
            'centro_global': centro_cara,
            'normal_local': normal_local,
            'normal_global': normal_global,
            'area': area,
            'vertices_cara': vertices_cara,
            'distancia_punto_eje': distancia_al_punto_eje,
            'normal_z_abs': abs(normal_global[2])
        })

    umbral_vertical = 0.7
    caras_laterales = [c for c in caras_info if c['normal_z_abs'] <= umbral_vertical]

    print(f"  Filtrado: {len(caras_laterales)} caras laterales de 6 totales")

    caras_laterales.sort(key=lambda x: x['distancia_punto_eje'], reverse=True)

    if len(caras_laterales) < 2:
        caras_laterales = caras_info
        caras_laterales.sort(key=lambda x: x['distancia_punto_eje'], reverse=True)

    dos_mas_alejadas = caras_laterales[:2]

    umbral_distancia_puntos = 0.1

    for cara in dos_mas_alejadas:
        centro_cara = cara['centro_global']
        normal_cara = cara['normal_global']
        vectores_puntos = puntos_limpios - centro_cara
        distancias_al_plano = np.abs(np.dot(vectores_puntos, normal_cara))
        puntos_cercanos = np.sum(distancias_al_plano < umbral_distancia_puntos)
        cara['puntos_cercanos'] = puntos_cercanos

    cara_exterior = max(dos_mas_alejadas, key=lambda x: x['puntos_cercanos'])

    print(f"  ✓ Cara seleccionada: {cara_exterior['nombre']} ({cara_exterior['puntos_cercanos']} puntos)")

    vec_desde_eje = cara_exterior['centro_global'] - punto_eje_ref
    dot_producto = np.dot(cara_exterior['normal_global'], vec_desde_eje)

    if dot_producto < 0:
        cara_exterior['normal_global'] = -cara_exterior['normal_global']

    cara_exterior['punto_eje_ref'] = punto_eje_ref
    cara_exterior['dot_producto'] = dot_producto

    return cara_exterior


def obtener_vertices_cara_por_nombre(nombre_cara, bb_info):
    """
    Obtiene los 4 vértices de una cara del paralelepípedo.
    (Código original de celda 3, sin modificaciones)
    """
    dims = bb_info['dimensiones']
    min_c = bb_info['min_coords']
    max_c = bb_info['max_coords']
    ejes = bb_info['ejes']
    centro_box = bb_info['centro']

    if nombre_cara == '+X':
        vertices_locales = np.array([
            [max_c[0], min_c[1], min_c[2]],
            [max_c[0], max_c[1], min_c[2]],
            [max_c[0], max_c[1], max_c[2]],
            [max_c[0], min_c[1], max_c[2]]
        ])
    elif nombre_cara == '-X':
        vertices_locales = np.array([
            [min_c[0], min_c[1], min_c[2]],
            [min_c[0], max_c[1], min_c[2]],
            [min_c[0], max_c[1], max_c[2]],
            [min_c[0], min_c[1], max_c[2]]
        ])
    elif nombre_cara == '+Y':
        vertices_locales = np.array([
            [min_c[0], max_c[1], min_c[2]],
            [max_c[0], max_c[1], min_c[2]],
            [max_c[0], max_c[1], max_c[2]],
            [min_c[0], max_c[1], max_c[2]]
        ])
    elif nombre_cara == '-Y':
        vertices_locales = np.array([
            [min_c[0], min_c[1], min_c[2]],
            [max_c[0], min_c[1], min_c[2]],
            [max_c[0], min_c[1], max_c[2]],
            [min_c[0], min_c[1], max_c[2]]
        ])
    elif nombre_cara == '+Z':
        vertices_locales = np.array([
            [min_c[0], min_c[1], max_c[2]],
            [max_c[0], min_c[1], max_c[2]],
            [max_c[0], max_c[1], max_c[2]],
            [min_c[0], max_c[1], max_c[2]]
        ])
    else:  # -Z
        vertices_locales = np.array([
            [min_c[0], min_c[1], min_c[2]],
            [max_c[0], min_c[1], min_c[2]],
            [max_c[0], max_c[1], min_c[2]],
            [min_c[0], max_c[1], min_c[2]]
        ])

    vertices_globales = centro_box + vertices_locales @ ejes.T
    return vertices_globales


# ============================================================================
# PROCESAMIENTO DE CADA EJEMPLAR - VERSIÓN MEJORADA
# ============================================================================

print("\n" + "=" * 60)
print("PROCESAMIENTO DE EJEMPLARES - CON SEPARACIÓN ANTENA/SOPORTE")
print("=" * 60)

# Configuración
METODO_OBB = 'comparar'  # Opciones: 'trimesh', 'open3d_pca', 'open3d_minimal', 'comparar'
ANCHO_VENTANA_DENSIDAD = 0.05  # 5 cm
MAX_DISTANCIA_ANALISIS = 0.5   # 50 cm a cada lado del plano
APLICAR_SEPARACION = True      # True para aplicar separación antena/soporte

resultados_ejemplares = {}

for ejemplar in ejemplares:
    print(f"\n{'='*60}")
    print(f"EJEMPLAR {int(ejemplar)}")
    print(f"{'='*60}")

    # Obtener puntos del ejemplar
    df_ejemplar = df_familia[df_familia['Ejemplar'] == ejemplar]
    puntos = df_ejemplar[['X_rel', 'Y_rel', 'Z_rel']].values

    print(f"  Puntos originales: {len(puntos)}")

    # 1. Limpieza de outliers
    puntos_limpios = limpiar_outliers(puntos)
    print(f"  Puntos tras limpieza: {len(puntos_limpios)} "
          f"(eliminados: {len(puntos) - len(puntos_limpios)})")

    # 2. NUEVO: Ajustar plano a la superficie frontal
    print("\n  [PASO 1] Ajuste de plano a superficie frontal:")
    try:
        plano_info = ajustar_plano_ransac(puntos_limpios,
                                          residual_threshold=0.05,
                                          max_trials=1000)
    except Exception as e:
        print(f"    ⚠️ Error ajustando plano: {e}")
        print(f"    Usando aproximación con centroide y PCA...")
        # Fallback: usar PCA simple
        centroide = np.mean(puntos_limpios, axis=0)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(puntos_limpios - centroide)
        normal = pca.components_[2]  # Tercer componente (menor varianza)
        plano_info = {
            'normal': normal,
            'punto_en_plano': centroide,
            'd': -np.dot(normal, centroide),
            'inliers_mask': np.ones(len(puntos_limpios), dtype=bool),
            'n_inliers': len(puntos_limpios)
        }

    # 3. NUEVO: Análisis de densidad perpendicular al plano
    if APLICAR_SEPARACION:
        print("\n  [PASO 2] Análisis de densidad perpendicular:")
        densidad_info = analizar_densidad_perpendicular(
            puntos_limpios,
            plano_info,
            ancho_ventana=ANCHO_VENTANA_DENSIDAD,
            max_distancia=MAX_DISTANCIA_ANALISIS
        )

        # 4. NUEVO: Separar antena de soporte
        print("\n  [PASO 3] Separación antena/soporte:")

        # Determinar en qué lado está la antena
        # (asumimos lado positivo por defecto, pero se puede ajustar)
        distancias = calcular_distancias_al_plano(puntos_limpios, plano_info)
        lado_con_mas_puntos = 'positivo' if np.sum(distancias > 0) > np.sum(distancias < 0) else 'negativo'
        print(f"    Lado con más puntos: {lado_con_mas_puntos}")

        separacion_info = separar_antena_soporte(
            puntos_limpios,
            plano_info,
            densidad_info,
            lado_antena=lado_con_mas_puntos,
            metodo='auto'
        )

        puntos_para_bbox = separacion_info['puntos_antena']

        print(f"\n  ✓ Usando {len(puntos_para_bbox)} puntos para calcular bbox final")
    else:
        puntos_para_bbox = puntos_limpios
        separacion_info = None
        print("\n  ℹ️ Separación antena/soporte desactivada")

    # 5. Calcular bounding box (con método seleccionado)
    print(f"\n  [PASO 4] Cálculo de Bounding Box (método: {METODO_OBB}):")
    bb_info = calcular_bounding_box(puntos_para_bbox, metodo=METODO_OBB)

    if bb_info is None:
        print(f"  ⚠️ No se pudo calcular bbox para ejemplar {int(ejemplar)}")
        continue

    print(f"\n  Bounding Box final ({bb_info['metodo']}):")
    print(f"    - Dimensiones: {bb_info['dimensiones'][0]:.3f} × "
          f"{bb_info['dimensiones'][1]:.3f} × {bb_info['dimensiones'][2]:.3f} m")
    print(f"    - Volumen: {bb_info['volumen']:.6f} m³")
    print(f"    - Puntos fuera: {bb_info['puntos_fuera']}")

    # 6. Identificar cara exterior
    print("\n  [PASO 5] Identificación de cara exterior:")
    cara_exterior = identificar_cara_exterior(bb_info, puntos_para_bbox)
    print(f"    Vector normal: ({cara_exterior['normal_global'][0]:.3f}, "
          f"{cara_exterior['normal_global'][1]:.3f}, "
          f"{cara_exterior['normal_global'][2]:.3f})")

    # Guardar resultados
    resultados_ejemplares[ejemplar] = {
        'puntos_originales': puntos,
        'puntos_limpios': puntos_limpios,
        'plano_info': plano_info,
        'densidad_info': densidad_info if APLICAR_SEPARACION else None,
        'separacion_info': separacion_info,
        'puntos_para_bbox': puntos_para_bbox,
        'bb_info': bb_info,
        'vertices': obtener_vertices_box(bb_info),
        'cara_exterior': cara_exterior
    }

print("\n" + "=" * 60)
print("PROCESAMIENTO COMPLETADO")
print("=" * 60)
print(f"\n✓ {len(resultados_ejemplares)} ejemplares procesados")
print("\nResultados disponibles en: resultados_ejemplares")
print("\nPara visualizar, usar el código de visualización de la celda 3 original")
