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


def ajustar_plano_bestfit(puntos):
    """
    Encuentra el plano de mejor ajuste a través de todos los puntos usando PCA.

    El plano se ajusta minimizando la suma de distancias cuadradas perpendiculares.
    La normal del plano es el eigenvector con menor eigenvalor (dirección de menor varianza).

    Como la mayoría de puntos forman la superficie reflectora de la antena,
    el plano se orientará automáticamente según esa superficie.

    Args:
        puntos: array Nx3 de puntos

    Returns:
        dict con:
        - normal: vector normal al plano (orientado hacia el exterior)
        - punto_en_plano: centroide (punto que pertenece al plano)
        - d: coeficiente d de la ecuación del plano
        - eigenvalues: valores propios ordenados (mayor a menor varianza)
        - n_inliers: número de puntos (todos se usan en PCA)
    """
    if len(puntos) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para ajustar un plano")

    # Calcular centroide
    centroide = np.mean(puntos, axis=0)

    # Centrar puntos
    puntos_centrados = puntos - centroide

    # PCA: calcular matriz de covarianza
    cov_matrix = np.cov(puntos_centrados.T)

    # Obtener eigenvalores y eigenvectores
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Ordenar por eigenvalor (mayor a menor varianza)
    idx_sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sorted].real  # Convertir a real
    eigenvectors = eigenvectors[:, idx_sorted].real

    # La normal es el eigenvector con MENOR eigenvalor (menor varianza)
    # Esto representa la dirección perpendicular al plano de mejor ajuste
    normal = eigenvectors[:, 2]  # Tercer componente (menor varianza)

    # Orientar la normal hacia el exterior (alejándose de la torre)
    # Vector desde torre (en el eje Z del centroide) hacia centroide
    punto_eje = np.array([0.0, 0.0, centroide[2]])
    vector_torre_centroide = centroide - punto_eje

    # Si la normal apunta hacia la torre, invertirla
    if np.dot(normal, vector_torre_centroide) < 0:
        normal = -normal

    # Coeficiente d de la ecuación ax + by + cz + d = 0
    d = -np.dot(normal, centroide)

    # Calcular ratio de varianza (mide qué tan plano es)
    ratio_varianza = eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 1e-10 else np.inf

    print(f"    Plano de mejor ajuste (PCA):")
    print(f"    - Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
    print(f"    - Eigenvalues: [{eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}, {eigenvalues[2]:.4f}]")
    print(f"    - Ratio varianza (max/min): {ratio_varianza:.1f}x")
    print(f"    - Todos los {len(puntos)} puntos usados en PCA")

    return {
        'normal': normal,
        'punto_en_plano': centroide,
        'd': d,
        'eigenvalues': eigenvalues,
        'n_inliers': len(puntos)  # Todos los puntos se usan
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


def detectar_punto_caida_densidad(densidad_info, lado='negativo', umbral_factor=0.3):
    """
    Detecta dónde TERMINA la cara interna de la antena buscando donde la densidad
    cae por debajo de un umbral basado en el máximo de densidad.

    Este método es más robusto que el percentil porque:
    - Identifica específicamente la transición desde la zona de alta densidad (antena)
      hacia la zona de baja densidad (espacio entre antena y soportes)
    - No asume una distribución específica de puntos

    Args:
        densidad_info: resultado de analizar_densidad_perpendicular
        lado: 'positivo' o 'negativo' (lado del plano a analizar)
        umbral_factor: factor para calcular umbral (default: 0.3 = 30% del máximo)

    Returns:
        dict con:
        - distancia_corte: distancia al plano donde se detecta la caída
        - densidad_corte: densidad en el punto de corte
        - umbral_usado: umbral de densidad aplicado
        - max_densidad: densidad máxima encontrada
        None si no se detecta
    """
    distancias = densidad_info['distancias']
    densidades = densidad_info['densidades']

    # Filtrar por lado
    if lado == 'positivo':
        mask = distancias > 0
    else:
        mask = distancias < 0

    dist_lado = distancias[mask]
    dens_lado = densidades[mask]

    if len(dens_lado) < 3:
        print(f"    ⚠️ Insuficientes puntos en lado {lado} para detectar caída de densidad")
        return None

    # Encontrar máximo de densidad en este lado
    max_densidad = np.max(dens_lado)
    umbral_densidad = umbral_factor * max_densidad

    print(f"    Análisis de caída de densidad en lado {lado}:")
    print(f"    - Densidad máxima: {max_densidad:.2f} puntos/m³")
    print(f"    - Umbral ({umbral_factor*100:.0f}%): {umbral_densidad:.2f} puntos/m³")

    # Buscar primer punto donde densidad cae por debajo del umbral
    # Empezar desde el plano (distancia ~0) y moverse hacia el interior
    if lado == 'negativo':
        # Ordenar de menos negativo (cerca del plano) a más negativo (hacia torre)
        indices_ordenados = np.argsort(dist_lado)[::-1]  # Descendente
    else:
        # Ordenar de menos positivo (cerca del plano) a más positivo (hacia fuera)
        indices_ordenados = np.argsort(dist_lado)

    dist_ordenado = dist_lado[indices_ordenados]
    dens_ordenado = dens_lado[indices_ordenados]

    # Buscar el primer punto donde la densidad cae por debajo del umbral
    indices_debajo = np.where(dens_ordenado < umbral_densidad)[0]

    if len(indices_debajo) > 0:
        idx_corte = indices_debajo[0]
        distancia_corte = dist_ordenado[idx_corte]
        densidad_corte = dens_ordenado[idx_corte]

        print(f"    ✓ Caída de densidad detectada:")
        print(f"      - Distancia al plano: {distancia_corte:.3f} m")
        print(f"      - Densidad en corte: {densidad_corte:.2f} puntos/m³")

        return {
            'distancia_corte': distancia_corte,
            'densidad_corte': densidad_corte,
            'umbral_usado': umbral_densidad,
            'max_densidad': max_densidad
        }
    else:
        print(f"    ⚠️ No se detectó caída de densidad significativa")
        print(f"      - Todas las densidades están por encima del umbral")
        return None


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
        # Los soportes están del lado OPUESTO a la antena
        # Si antena está en lado positivo → soportes en lado negativo
        # Si antena está en lado negativo → soportes en lado positivo
        lado_soporte = 'negativo' if lado_antena == 'positivo' else 'positivo'

        print(f"    Buscando soportes en lado {lado_soporte}")

        # MÉTODO 1 (NUEVO): Detectar caída de densidad en cara interna
        print(f"\n    [Método 1] Detección por caída de densidad:")
        caida_info = detectar_punto_caida_densidad(
            densidad_info,
            lado=lado_soporte,
            umbral_factor=0.3  # 30% del máximo
        )

        if caida_info is not None:
            distancia_corte = caida_info['distancia_corte']
            print(f"    ✓ Usando corte por caída de densidad: {distancia_corte:.3f} m")
        else:
            # MÉTODO 2: Detectar cambio brusco con derivada
            print(f"\n    [Método 2] Detección por derivada (fallback):")
            cambio_info = detectar_cambio_brusco_densidad(
                densidad_info,
                lado=lado_soporte
            )

            if cambio_info is not None:
                distancia_corte = cambio_info['distancia_corte']
                print(f"    ✓ Usando corte por derivada: {distancia_corte:.3f} m")
            else:
                # MÉTODO 3: Fallback final con percentil
                print(f"\n    [Método 3] Percentil (fallback final):")
                if lado_antena == 'positivo':
                    # Antena en lado positivo → soportes en lado negativo
                    # Usar percentil 25 de los negativos (los más cercanos al plano)
                    distancia_corte = np.percentile(distancias[distancias < 0], 25)
                else:
                    # Antena en lado negativo → soportes en lado positivo
                    # Usar percentil 75 de los positivos (los más cercanos al plano)
                    distancia_corte = np.percentile(distancias[distancias > 0], 75)
                print(f"    ℹ️ Usando corte por percentil: {distancia_corte:.3f} m")

    elif metodo == 'umbral_fijo':
        # Usar umbral fijo en el lado de los SOPORTES
        # Si antena en lado positivo → soportes en negativo → umbral negativo
        distancia_corte = -0.15 if lado_antena == 'positivo' else 0.15
        print(f"    ℹ️ Usando umbral fijo en lado soportes: {distancia_corte:.3f} m")

    else:
        raise ValueError(f"Método '{metodo}' no reconocido")

    # Aplicar corte
    # El punto de corte detectado marca dónde TERMINA la antena y EMPIEZA el soporte
    # Si lado_antena='positivo': antena en lado positivo, soportes en lado negativo
    #   - distancia_corte es negativa (ej: -0.125m)
    #   - Soportes: distancias < distancia_corte (ej: < -0.125m, más hacia la torre)
    #   - Antena: distancias >= distancia_corte (ej: >= -0.125m, incluye todo el lado positivo)
    if lado_antena == 'positivo':
        # Antena está en el lado positivo (exterior)
        # Corte detectado en lado negativo (hacia torre) donde empiezan soportes
        # Mantener todo lo que está MÁS AFUERA que el corte (distancias >= distancia_corte)
        mask_antena = distancias >= distancia_corte
    else:
        # Antena está en el lado negativo (raro, pero por completitud)
        # Corte detectado en lado positivo donde empiezan soportes
        # Mantener todo lo que está MÁS ADENTRO que el corte (distancias <= distancia_corte)
        mask_antena = distancias <= distancia_corte

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

    # 2. NUEVO: Ajustar plano de mejor ajuste a la superficie frontal
    print("\n  [PASO 1] Ajuste de plano a superficie frontal:")
    plano_info = ajustar_plano_bestfit(puntos_limpios)

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

        # Determinar en qué lado buscar el soporte
        # El plano PCA tiene normal radial apuntando hacia AFUERA desde la torre
        # Por tanto:
        #   - Lado POSITIVO (distancias > 0): exterior, alejándose de la torre → ANTENA
        #   - Lado NEGATIVO (distancias < 0): interior, hacia la torre → SOPORTES

        # Los soportes están DETRÁS de la antena, hacia la torre (lado negativo)
        # Así que buscamos el cambio brusco en el lado NEGATIVO
        distancias = calcular_distancias_al_plano(puntos_limpios, plano_info)
        n_positivos = np.sum(distancias > 0)
        n_negativos = np.sum(distancias < 0)

        print(f"    Distribución de puntos:")
        print(f"    - Lado positivo (exterior): {n_positivos} puntos ({100*n_positivos/len(distancias):.1f}%)")
        print(f"    - Lado negativo (interior/torre): {n_negativos} puntos ({100*n_negativos/len(distancias):.1f}%)")

        # Buscar soportes en el lado NEGATIVO (hacia la torre)
        lado_buscar_soporte = 'negativo'
        print(f"    Buscando soportes en lado: {lado_buscar_soporte}")

        separacion_info = separar_antena_soporte(
            puntos_limpios,
            plano_info,
            densidad_info,
            lado_antena='positivo',  # La antena está en el lado POSITIVO (exterior)
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


# ============================================================================
# EXPORTAR RESULTADOS A EXCEL
# ============================================================================

print("\n" + "=" * 60)
print("EXPORTANDO RESULTADOS A EXCEL")
print("=" * 60)

try:
    import pandas as pd
    from datetime import datetime

    # Crear lista de resultados para DataFrame
    datos_excel = []

    for ejemplar, datos in resultados_ejemplares.items():
        bb_info = datos['bb_info']
        cara = datos['cara_exterior']
        plano_info = datos['plano_info']
        separacion_info = datos['separacion_info']

        # Preparar datos del ejemplar
        fila = {
            'Familia': int(familia_seleccionada),
            'Ejemplar': int(ejemplar),
            'Puntos_Originales': len(datos['puntos_originales']),
            'Puntos_Limpios': len(datos['puntos_limpios']),
            'Puntos_Antena': len(datos['puntos_para_bbox']),
        }

        # Información de separación
        if separacion_info:
            fila['Puntos_Soporte'] = len(separacion_info['puntos_soporte'])
            fila['Porcentaje_Soporte'] = 100 * len(separacion_info['puntos_soporte']) / len(datos['puntos_limpios'])
            fila['Distancia_Corte_m'] = separacion_info['distancia_corte']
        else:
            fila['Puntos_Soporte'] = 0
            fila['Porcentaje_Soporte'] = 0
            fila['Distancia_Corte_m'] = None

        # Dimensiones del Bounding Box
        fila['BB_Dim_X_m'] = bb_info['dimensiones'][0]
        fila['BB_Dim_Y_m'] = bb_info['dimensiones'][1]
        fila['BB_Dim_Z_m'] = bb_info['dimensiones'][2]
        fila['BB_Volumen_m3'] = bb_info['volumen']
        fila['BB_Puntos_Fuera'] = bb_info['puntos_fuera']
        fila['BB_Metodo'] = bb_info['metodo']

        # Centro del Bounding Box
        fila['BB_Centro_X'] = bb_info['centro'][0]
        fila['BB_Centro_Y'] = bb_info['centro'][1]
        fila['BB_Centro_Z'] = bb_info['centro'][2]

        # Plano de mejor ajuste (PCA)
        fila['Plano_Normal_X'] = plano_info['normal'][0]
        fila['Plano_Normal_Y'] = plano_info['normal'][1]
        fila['Plano_Normal_Z'] = plano_info['normal'][2]
        fila['Plano_Eigenvalue_Max'] = plano_info['eigenvalues'][0]
        fila['Plano_Eigenvalue_Med'] = plano_info['eigenvalues'][1]
        fila['Plano_Eigenvalue_Min'] = plano_info['eigenvalues'][2]

        # Cara exterior
        fila['Cara_Exterior_Nombre'] = cara['nombre']
        fila['Cara_Normal_X'] = cara['normal_global'][0]
        fila['Cara_Normal_Y'] = cara['normal_global'][1]
        fila['Cara_Normal_Z'] = cara['normal_global'][2]
        fila['Cara_Centro_X'] = cara['centro_global'][0]
        fila['Cara_Centro_Y'] = cara['centro_global'][1]
        fila['Cara_Centro_Z'] = cara['centro_global'][2]
        fila['Cara_Area_m2'] = cara['area']
        fila['Cara_Puntos_Cercanos'] = cara['puntos_cercanos']
        fila['Cara_Dist_Punto_Eje_m'] = cara['distancia_punto_eje']

        datos_excel.append(fila)

    # Crear DataFrame
    df_resultados = pd.DataFrame(datos_excel)

    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"resultados_familia_{int(familia_seleccionada)}_{timestamp}.xlsx"

    # Exportar a Excel con formato
    with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
        # Hoja principal con todos los resultados
        df_resultados.to_excel(writer, sheet_name='Resultados', index=False)

        # Ajustar anchos de columna
        worksheet = writer.sheets['Resultados']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Crear hoja de resumen
        resumen_data = {
            'Métrica': [
                'Familia',
                'Número de ejemplares',
                'Total puntos originales',
                'Total puntos tras limpieza',
                'Total puntos antena',
                'Total puntos soporte',
                'Porcentaje medio soporte eliminado (%)',
                'Método OBB',
                'Volumen medio bounding box (m³)',
                'Dimensión X media (m)',
                'Dimensión Y media (m)',
                'Dimensión Z media (m)'
            ],
            'Valor': [
                int(familia_seleccionada),
                len(resultados_ejemplares),
                df_resultados['Puntos_Originales'].sum(),
                df_resultados['Puntos_Limpios'].sum(),
                df_resultados['Puntos_Antena'].sum(),
                df_resultados['Puntos_Soporte'].sum(),
                df_resultados['Porcentaje_Soporte'].mean(),
                METODO_OBB,
                df_resultados['BB_Volumen_m3'].mean(),
                df_resultados['BB_Dim_X_m'].mean(),
                df_resultados['BB_Dim_Y_m'].mean(),
                df_resultados['BB_Dim_Z_m'].mean()
            ]
        }
        df_resumen = pd.DataFrame(resumen_data)
        df_resumen.to_excel(writer, sheet_name='Resumen', index=False)

        # Ajustar anchos en hoja resumen
        worksheet_resumen = writer.sheets['Resumen']
        worksheet_resumen.column_dimensions['A'].width = 40
        worksheet_resumen.column_dimensions['B'].width = 25

    print(f"✓ Resultados exportados a: {nombre_archivo}")
    print(f"  - Hoja 'Resultados': {len(df_resultados)} ejemplares con {len(df_resultados.columns)} columnas")
    print(f"  - Hoja 'Resumen': Estadísticas globales de la familia")

except ImportError:
    print("⚠️ pandas u openpyxl no están instalados. Instalar con:")
    print("   pip install pandas openpyxl")
except Exception as e:
    print(f"⚠️ Error al exportar a Excel: {e}")


# ============================================================================
# VISUALIZACIONES DE DEBUG - NUEVAS (Separación Antena/Soporte)
# ============================================================================

if APLICAR_SEPARACION:
    print("\n" + "=" * 60)
    print("VISUALIZACIONES DE DEBUG - ANÁLISIS ANTENA/SOPORTE")
    print("=" * 60)

    # Calcular número de filas y columnas para subplots de debug
    n_ejemplares = len(ejemplares)
    n_cols_debug = min(3, n_ejemplares)
    n_rows_debug = int(np.ceil(n_ejemplares / n_cols_debug))

    # ========================================================================
    # 1. VISUALIZACIÓN DEL PLANO AJUSTADO Y SEPARACIÓN DE PUNTOS
    # ========================================================================

    fig_plano, axes_plano = plt.subplots(n_rows_debug, n_cols_debug,
                                         figsize=(7*n_cols_debug, 7*n_rows_debug))
    if n_ejemplares == 1:
        axes_plano = np.array([axes_plano])
    axes_plano = axes_plano.flatten()

    for idx, (ejemplar, datos) in enumerate(resultados_ejemplares.items()):
        ax = axes_plano[idx]

        if datos['separacion_info'] is None:
            ax.text(0.5, 0.5, 'Separación no aplicada',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue

        plano_info = datos['plano_info']
        separacion_info = datos['separacion_info']
        puntos_antena = separacion_info['puntos_antena']
        puntos_soporte = separacion_info['puntos_soporte']

        # Puntos coloreados: verde=antena, rojo=soporte
        ax.scatter(puntos_antena[:, 0], puntos_antena[:, 1],
                  c='green', s=40, alpha=0.7, edgecolors='black',
                  linewidth=0.3, label=f'Antena ({len(puntos_antena)})')

        if len(puntos_soporte) > 0:
            ax.scatter(puntos_soporte[:, 0], puntos_soporte[:, 1],
                      c='red', s=40, alpha=0.5, edgecolors='black',
                      linewidth=0.3, label=f'Soporte ({len(puntos_soporte)})')

        # Dibujar proyección del plano ajustado
        normal = plano_info['normal']
        punto_plano = plano_info['punto_en_plano']

        # Crear una cuadrícula para visualizar el plano
        all_pts = datos['puntos_limpios']
        x_range = [all_pts[:, 0].min(), all_pts[:, 0].max()]
        y_range = [all_pts[:, 1].min(), all_pts[:, 1].max()]

        # Dibujar línea de intersección del plano con Z=promedio
        # El plano es: normal·(P - punto_plano) = 0
        # En 2D, mostrar la proyección como una línea

        # Centro del plano en proyección XY
        ax.plot(punto_plano[0], punto_plano[1], 'o', color='blue',
               markersize=12, markeredgecolor='white', markeredgewidth=2,
               label='Centro plano', zorder=10)

        # Vector normal proyectado en XY
        normal_xy = normal[:2] / (np.linalg.norm(normal[:2]) + 1e-10)
        vector_length = 0.3
        ax.arrow(punto_plano[0], punto_plano[1],
                normal_xy[0]*vector_length, normal_xy[1]*vector_length,
                head_width=0.05, head_length=0.05, fc='blue', ec='blue',
                linewidth=2, alpha=0.7, label='Normal (XY)', zorder=10)

        # Línea del corte
        distancia_corte = separacion_info['distancia_corte']
        # Calcular punto del corte en la dirección de la normal
        punto_corte = punto_plano + normal * distancia_corte

        ax.plot(punto_corte[0], punto_corte[1], 'X', color='orange',
               markersize=15, markeredgecolor='black', markeredgewidth=2,
               label=f'Corte ({distancia_corte:.3f}m)', zorder=10)

        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.set_title(f'Ejemplar {int(ejemplar)} - Separación Antena/Soporte\n'
                    f'{len(puntos_antena)} pts antena, {len(puntos_soporte)} pts soporte',
                    fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)

    # Ocultar ejes vacíos
    for idx in range(n_ejemplares, len(axes_plano)):
        axes_plano[idx].axis('off')

    plt.tight_layout()
    plt.show()

    print("✓ Visualización de separación antena/soporte completada")

    # ========================================================================
    # 2. GRÁFICOS DE DENSIDAD vs DISTANCIA CON DERIVADA
    # ========================================================================

    fig_dens, axes_dens = plt.subplots(n_rows_debug, n_cols_debug,
                                       figsize=(7*n_cols_debug, 6*n_rows_debug))
    if n_ejemplares == 1:
        axes_dens = np.array([axes_dens])
    axes_dens = axes_dens.flatten()

    for idx, (ejemplar, datos) in enumerate(resultados_ejemplares.items()):
        ax = axes_dens[idx]

        if datos['densidad_info'] is None:
            ax.text(0.5, 0.5, 'Análisis de densidad no aplicado',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue

        densidad_info = datos['densidad_info']
        separacion_info = datos['separacion_info']

        distancias = densidad_info['distancias']
        densidades = densidad_info['densidades']

        # Gráfico de densidad
        ax.plot(distancias, densidades, 'b-o', linewidth=2, markersize=5,
               label='Densidad (puntos/m³)', alpha=0.7)

        # Línea del plano (distancia = 0)
        ax.axvline(0, color='blue', linestyle='--', linewidth=2,
                  alpha=0.5, label='Plano frontal')

        # Línea del corte detectado
        if separacion_info:
            distancia_corte = separacion_info['distancia_corte']
            ax.axvline(distancia_corte, color='red', linestyle='--',
                      linewidth=2.5, alpha=0.8,
                      label=f'Corte ({distancia_corte:.3f}m)')

            # Sombrear zona de antena
            if distancia_corte > 0:
                ax.axvspan(0, distancia_corte, alpha=0.15, color='green',
                          label='Zona antena')
            else:
                ax.axvspan(distancia_corte, 0, alpha=0.15, color='green',
                          label='Zona antena')

        ax.set_xlabel('Distancia al plano (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Densidad (puntos/m³)', fontsize=11, fontweight='bold')
        ax.set_title(f'Ejemplar {int(ejemplar)} - Análisis de Densidad',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)

    # Ocultar ejes vacíos
    for idx in range(n_ejemplares, len(axes_dens)):
        axes_dens[idx].axis('off')

    plt.tight_layout()
    plt.show()

    print("✓ Visualización de densidad vs distancia completada")


# ============================================================================
# VISUALIZACIÓN EN PLANTA - VISTA GENERAL (Visualizaciones Originales)
# ============================================================================

print("\n" + "=" * 60)
print("GENERANDO VISUALIZACIÓN EN PLANTA - VISTA GENERAL")
print("=" * 60)
print("\nNOTA: El Bounding Box es un PARALELEPÍPEDO 3D orientado (OBB)")
print(f"      calculado con método: {METODO_OBB}")
print("")
print("ALGORITMO MEJORADO de identificación de cara exterior:")
print("  1. Se filtran las caras LATERALES (excluir superior/inferior con |normal.z| > 0.7)")
print("  2. Se ordenan por distancia al punto del eje (0, 0, z_centro)")
print("  3. Se toman las 2 caras más alejadas")
print("  4. Se cuentan los puntos de la nube cercanos a cada cara (< 10cm)")
print("  5. Se elige la cara con MÁS PUNTOS cercanos")
print("  6. El vector normal apunta desde el centro de esa cara hacia fuera")
print("")
print("La vista en planta muestra:")
print("  - Aristas de la base (línea sólida)")
print("  - Aristas del techo (línea discontinua)")
print("  - Aristas verticales (línea punteada)")
print("  - Cara exterior en ROJO (la más alejada con más puntos)")
print("  - Punto del eje de referencia en NARANJA (0, 0, z_centro)")
print("  - Vector naranja: dirección desde punto eje → centro cara")
print("  - Vector rojo: normal perpendicular a la cara exterior")
print("")

fig, ax = plt.subplots(figsize=(14, 14))

# Colores para cada ejemplar
colores = plt.cm.Set3(np.linspace(0, 1, len(ejemplares)))

for idx, (ejemplar, datos) in enumerate(resultados_ejemplares.items()):
    color = colores[idx]

    # Obtener los 8 vértices del bounding box 3D
    vertices = datos['vertices']  # 8 vértices [x, y, z]

    # Proyección en XY (vista en planta)
    vertices_xy = vertices[:, :2]

    # Dibujar todas las aristas del paralelepípedo proyectadas en planta
    # Los vértices están ordenados así:
    # 0-3: base inferior, 4-7: techo superior
    # Conexiones: 0-1, 1-2, 2-3, 3-0 (base)
    #            4-5, 5-6, 6-7, 7-4 (techo)
    #            0-4, 1-5, 2-6, 3-7 (verticales)

    aristas = [
        # Base inferior
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Techo superior
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Aristas verticales
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Dibujar todas las aristas
    for i, arista in enumerate(aristas):
        v1, v2 = arista
        if i < 4:  # Base
            ax.plot([vertices_xy[v1, 0], vertices_xy[v2, 0]],
                   [vertices_xy[v1, 1], vertices_xy[v2, 1]],
                   color=color, linewidth=2.5, alpha=0.8)
        elif i < 8:  # Techo
            ax.plot([vertices_xy[v1, 0], vertices_xy[v2, 0]],
                   [vertices_xy[v1, 1], vertices_xy[v2, 1]],
                   color=color, linewidth=2.5, alpha=0.6, linestyle='--')
        else:  # Verticales
            ax.plot([vertices_xy[v1, 0], vertices_xy[v2, 0]],
                   [vertices_xy[v1, 1], vertices_xy[v2, 1]],
                   color=color, linewidth=1.5, alpha=0.5, linestyle=':')

    # Rellenar el polígono convexo de todos los vértices proyectados
    hull = ConvexHull(vertices_xy)
    vertices_ordenados = vertices_xy[hull.vertices]
    polygon = plt.Polygon(vertices_ordenados, fill=True, alpha=0.15,
                         facecolor=color, edgecolor='none',
                         label=f'Ejemplar {int(ejemplar)}')
    ax.add_patch(polygon)

    # ===== CARA EXTERIOR =====
    cara = datos['cara_exterior']
    centro_cara = cara['centro_global']
    normal = cara['normal_global']
    vertices_cara_3d = cara['vertices_cara']  # Ya calculados en identificar_cara_exterior

    # Proyectar vértices de la cara en XY
    vertices_cara_xy = vertices_cara_3d[:, :2]

    # Dibujar la cara exterior rellena
    cara_poly = plt.Polygon(vertices_cara_xy, fill=True, alpha=0.4,
                           facecolor='red', edgecolor='darkred', linewidth=4,
                           zorder=5)
    ax.add_patch(cara_poly)

    # ===== CENTRO DE LA CARA EXTERIOR =====
    ax.plot(centro_cara[0], centro_cara[1], 'o', color='darkred',
           markersize=14, markeredgecolor='white', markeredgewidth=2.5,
           zorder=8, label='Centro cara ext.' if idx == 0 else '')

    # Centro del bounding box
    centro = datos['bb_info']['centro']
    ax.plot(centro[0], centro[1], 'o', color=color, markersize=12,
           markeredgecolor='black', markeredgewidth=2, zorder=6)

    # ===== PUNTO DEL EJE DE REFERENCIA (0,0,z) =====
    punto_eje = cara['punto_eje_ref']
    ax.plot(punto_eje[0], punto_eje[1], 's', color='orange', markersize=12,
           markeredgecolor='black', markeredgewidth=2, zorder=7,
           label='Punto eje (0,0,z)' if idx == 0 else '')

    # ===== VECTOR: PUNTO EJE → CENTRO CARA (dirección) =====
    arrow_eje_cara = FancyArrowPatch(
        (punto_eje[0], punto_eje[1]),
        (centro_cara[0], centro_cara[1]),
        arrowstyle='->', mutation_scale=25, linewidth=2.5,
        color='orange', alpha=0.8, zorder=6,
        label='Vector eje→cara' if idx == 0 else ''
    )
    ax.add_patch(arrow_eje_cara)

    # ===== VECTOR NORMAL desde centro de cara =====
    escala_vector = 1.0
    vector_end = centro_cara[:2] + normal[:2] * escala_vector

    arrow = FancyArrowPatch(
        (centro_cara[0], centro_cara[1]),
        (vector_end[0], vector_end[1]),
        arrowstyle='->', mutation_scale=25, linewidth=3,
        color='red', zorder=10,
        label='Vector normal' if idx == 0 else ''
    )
    ax.add_patch(arrow)

    # Etiqueta del ejemplar
    ax.text(centro[0], centro[1], f'E{int(ejemplar)}',
           fontsize=13, fontweight='bold', ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                    edgecolor=color, linewidth=2))

# Dibujar el eje de la torre (origen)
ax.plot(0, 0, 'X', color='black', markersize=22, markeredgecolor='red',
       markeredgewidth=3, label='Eje Torre (0,0)', zorder=10)

# Círculo que representa la torre
circulo_torre = plt.Circle((0, 0), 0.5, fill=False, edgecolor='black',
                          linewidth=2.5, linestyle='--', label='Torre')
ax.add_patch(circulo_torre)

# Configuración de ejes
ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
ax.set_title(f'Vista en Planta - Familia {int(familia_seleccionada)}\n'
            f'Paralelepípedos 3D Orientados (OBB) - Proyección en XY\n'
            f'(Cara Exterior = Rojo Oscuro | Base = Sólida | Techo = Discontinua | Verticales = Punteada)',
            fontsize=15, fontweight='bold', pad=20)

# Aspecto igual para mantener proporciones
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

# Crear elementos de leyenda personalizados
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
elementos_leyenda = [
    Line2D([0], [0], color='gray', linewidth=2.5, label='Aristas Base (sólida)'),
    Line2D([0], [0], color='gray', linewidth=2.5, linestyle='--', label='Aristas Techo (discontinua)'),
    Line2D([0], [0], color='gray', linewidth=1.5, linestyle=':', label='Aristas Verticales (punteada)'),
    Patch(facecolor='red', alpha=0.4, edgecolor='darkred', linewidth=4, label='Cara Exterior (rellena)'),
    Line2D([0], [0], marker='o', color='darkred', markersize=12, linestyle='None',
           markeredgecolor='white', markeredgewidth=2, label='Centro cara exterior'),
    Line2D([0], [0], marker='X', color='black', markersize=12, linestyle='None',
           markeredgecolor='red', markeredgewidth=2, label='Eje Torre (0,0,0)'),
    Line2D([0], [0], marker='s', color='orange', markersize=10, linestyle='None',
           markeredgecolor='black', markeredgewidth=2, label='Punto Eje (0,0,z)'),
    FancyArrowPatch((0,0), (0.1,0.1), arrowstyle='->', mutation_scale=15,
                    linewidth=2, color='orange', label='Vector eje→cara'),
    FancyArrowPatch((0,0), (0.1,0.1), arrowstyle='->', mutation_scale=15,
                    linewidth=2.5, color='red', label='Vector Normal')
]

# Añadir ejemplares a la leyenda
for idx, ejemplar in enumerate(ejemplares):
    elementos_leyenda.append(
        Line2D([0], [0], marker='o', color=colores[idx], markersize=10,
               linestyle='None', markeredgecolor='black', markeredgewidth=1.5,
               label=f'Ejemplar {int(ejemplar)}')
    )

ax.legend(handles=elementos_leyenda, loc='upper left', fontsize=9, framealpha=0.95, ncol=2)

# Ajustar límites
all_points = np.vstack([datos['puntos_limpios'] for datos in resultados_ejemplares.values()])
margin = 2
ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)

plt.tight_layout()
plt.show()

print("✓ Visualización general completada")


# ============================================================================
# VISUALIZACIÓN EN PLANTA - EJEMPLARES INDIVIDUALES
# ============================================================================

print("\n" + "=" * 60)
print("GENERANDO VISUALIZACIONES INDIVIDUALES POR EJEMPLAR")
print("=" * 60)

# Calcular número de filas y columnas para subplots
n_ejemplares = len(ejemplares)
n_cols = min(3, n_ejemplares)
n_rows = int(np.ceil(n_ejemplares / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
if n_ejemplares == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, (ejemplar, datos) in enumerate(resultados_ejemplares.items()):
    ax = axes[idx]
    color = colores[idx]

    # Puntos para bbox (puede ser filtrados o todos)
    puntos = datos['puntos_para_bbox']
    ax.scatter(puntos[:, 0], puntos[:, 1], c=[color], s=40, alpha=0.7,
              edgecolors='black', linewidth=0.5, label='Puntos')

    # Obtener los 8 vértices del bounding box
    vertices = datos['vertices']
    vertices_xy = vertices[:, :2]

    # Dibujar todas las aristas del paralelepípedo
    aristas = [
        # Base inferior
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Techo superior
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Aristas verticales
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    for i, arista in enumerate(aristas):
        v1, v2 = arista
        if i < 4:  # Base
            ax.plot([vertices_xy[v1, 0], vertices_xy[v2, 0]],
                   [vertices_xy[v1, 1], vertices_xy[v2, 1]],
                   color=color, linewidth=2.5, alpha=0.8,
                   label='Base' if i == 0 else '')
        elif i < 8:  # Techo
            ax.plot([vertices_xy[v1, 0], vertices_xy[v2, 0]],
                   [vertices_xy[v1, 1], vertices_xy[v2, 1]],
                   color=color, linewidth=2.5, alpha=0.6, linestyle='--',
                   label='Techo' if i == 4 else '')
        else:  # Verticales
            ax.plot([vertices_xy[v1, 0], vertices_xy[v2, 0]],
                   [vertices_xy[v1, 1], vertices_xy[v2, 1]],
                   color=color, linewidth=1.5, alpha=0.5, linestyle=':',
                   label='Aristas verticales' if i == 8 else '')

    # ===== CARA EXTERIOR =====
    cara = datos['cara_exterior']
    centro_cara = cara['centro_global']
    normal = cara['normal_global']
    vertices_cara_3d = cara['vertices_cara']

    # Proyectar vértices en XY
    vertices_cara_xy = vertices_cara_3d[:, :2]

    # Rellenar cara exterior
    cara_poly = plt.Polygon(vertices_cara_xy, fill=True, alpha=0.3,
                           facecolor='red', edgecolor='darkred', linewidth=4,
                           label='Cara Exterior')
    ax.add_patch(cara_poly)

    # Centro de la cara exterior
    ax.plot(centro_cara[0], centro_cara[1], 'o', color='darkred',
           markersize=14, markeredgecolor='white', markeredgewidth=2.5,
           label='Centro cara ext.', zorder=8)

    # Centro del BB
    centro = datos['bb_info']['centro']
    ax.plot(centro[0], centro[1], 'o', color=color, markersize=15,
           markeredgecolor='black', markeredgewidth=2, label='Centro BB', zorder=6)

    # Punto del eje a la altura del centro
    punto_eje = cara['punto_eje_ref']
    ax.plot(punto_eje[0], punto_eje[1], 's', color='orange', markersize=12,
           markeredgecolor='black', markeredgewidth=2, label='Punto Eje (0,0,z)', zorder=7)

    # Vector desde punto eje al centro de la cara
    arrow_eje_cara = FancyArrowPatch(
        (punto_eje[0], punto_eje[1]),
        (centro_cara[0], centro_cara[1]),
        arrowstyle='->', mutation_scale=20, linewidth=2,
        color='orange', alpha=0.8, zorder=6,
        label='Vector eje→cara'
    )
    ax.add_patch(arrow_eje_cara)

    # Vector normal
    escala_vector = max(datos['bb_info']['dimensiones'][:2]) * 0.5
    vector_end = centro_cara[:2] + normal[:2] * escala_vector

    arrow = FancyArrowPatch(
        (centro_cara[0], centro_cara[1]),
        (vector_end[0], vector_end[1]),
        arrowstyle='->', mutation_scale=25, linewidth=3,
        color='red', zorder=10, label='Vector Normal'
    )
    ax.add_patch(arrow)

    # Eje de la torre
    ax.plot(0, 0, 'X', color='black', markersize=20, markeredgecolor='red',
           markeredgewidth=2.5, zorder=10)

    # Configuración
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Ejemplar {int(ejemplar)}\n({len(puntos)} puntos)',
                fontsize=14, fontweight='bold', pad=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)

    # Ajustar límites con margen
    margin = max(datos['bb_info']['dimensiones'][:2]) * 0.3
    ax.set_xlim(puntos[:, 0].min() - margin, puntos[:, 0].max() + margin)
    ax.set_ylim(puntos[:, 1].min() - margin, puntos[:, 1].max() + margin)

# Ocultar ejes vacíos si hay menos ejemplares que subplots
for idx in range(n_ejemplares, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

print(f"✓ {n_ejemplares} visualizaciones individuales completadas")


# ============================================================================
# VISUALIZACIÓN 3D DE BOUNDING BOXES
# ============================================================================

print("\nGenerando visualización 3D de bounding boxes...")

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

for idx, (ejemplar, datos) in enumerate(resultados_ejemplares.items()):
    color = colores[idx]

    # Puntos para bbox
    puntos = datos['puntos_para_bbox']
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2],
              c=[color], s=30, alpha=0.6, edgecolors='black', linewidth=0.3,
              label=f'Ejemplar {int(ejemplar)}')

    # Dibujar bounding box
    vertices = datos['vertices']

    # Definir las 6 caras del cuboide
    caras = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Base
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Techo
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Cara 1
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Cara 2
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Cara 3
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # Cara 4
    ]

    # Dibujar caras del bounding box
    poly = Poly3DCollection(caras, alpha=0.15, facecolor=color,
                           edgecolor='black', linewidth=1.5)
    ax.add_collection3d(poly)

    # Vector normal desde la cara exterior
    cara = datos['cara_exterior']
    centro_cara = cara['centro_global']
    normal = cara['normal_global']

    # Dibujar vector normal
    escala_vector = 1.0
    ax.quiver(centro_cara[0], centro_cara[1], centro_cara[2],
             normal[0]*escala_vector, normal[1]*escala_vector, normal[2]*escala_vector,
             color='red', arrow_length_ratio=0.3, linewidth=3)

    # Punto del eje a la altura del centro de la BB
    punto_eje = cara['punto_eje_ref']
    ax.scatter([punto_eje[0]], [punto_eje[1]], [punto_eje[2]],
              c='orange', marker='s', s=200,
              label='Punto eje ref' if idx == 0 else '',
              edgecolors='black', linewidth=2, zorder=8)

    # Línea desde punto del eje al centro de la cara
    ax.plot([punto_eje[0], centro_cara[0]],
           [punto_eje[1], centro_cara[1]],
           [punto_eje[2], centro_cara[2]],
           color='orange', linewidth=2, linestyle=':', alpha=0.7, zorder=7)

# Eje de la torre
ax.scatter([0], [0], [0], c='black', marker='X', s=300,
          label='Eje Torre', edgecolors='red', linewidth=2)

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
ax.set_title(f'Vista 3D - Familia {int(familia_seleccionada)} con Bounding Boxes',
            fontsize=14, fontweight='bold')

# Aspecto igual
max_range = np.array([
    all_points[:, 0].max() - all_points[:, 0].min(),
    all_points[:, 1].max() - all_points[:, 1].min(),
    all_points[:, 2].max() - all_points[:, 2].min()
]).max() / 2.0

mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_box_aspect([1, 1, 1])

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualización 3D completada")

print("\n" + "=" * 60)
print("TODAS LAS VISUALIZACIONES COMPLETADAS")
print("=" * 60)
