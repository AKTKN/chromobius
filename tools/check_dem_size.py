import argparse
import pathlib
import sys
from typing import List, NamedTuple

project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from beliefmatching.belief_matching import detector_error_model_to_check_matrices
from relay.tests.testdata.utils import filter_detectors_by_basis

try:
    import stim
except ImportError:
    print("The 'stim' library is required to run this script. Please install it via 'pip install stim'.")
    sys.exit(1)

"""
Usage(see tools/gen_circuits):
    python3 tools/gen_circuits     
    --out_dir ./generated_circuits     
    --diameter 21     
    --rounds "d*1"     
    --noise_strength 0.001     
    --noise_model si1000     
    --style midout_color_code_Z

python3 check_dem_size.py --circuits_dir ./generated_circuits
"""


class DemSizeInfo(NamedTuple):
    "Data class to hold DEM size information for a circuit."
    filepath: pathlib.Path
    num_detectors: int
    num_observables: int
    num_errors: int
    num_instructions: int

def analyze_stim_circuit(circuit_path: pathlib.Path) -> DemSizeInfo:
    "Read a stim circuit and extract DEM information"

    if not circuit_path.is_file():
        raise FileNotFoundError(f"Circuit file not found: {circuit_path}")
    
    try:
        circuit = stim.Circuit.from_file(circuit_path)
        dem = circuit.detector_error_model()

        return DemSizeInfo(
            filepath=circuit_path,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
            num_errors=dem.num_errors,
            num_instructions=len(circuit)
        )
    except Exception as e:
        raise ValueError(f"Failed to analyze circuit {circuit_path}: {e}") from e
    
import numpy as np
from scipy.sparse import csc_matrix, hstack, isspmatrix

def compress_dem(dem_matrix, priors):
    """
    IBMのRelay-BP論文で説明されているDEM（H行列）の圧縮処理を行います。
    

    H行列内で同一の列（同じシンドロームを引き起こすエラー）を検出し、
    それらを1つの列にマージします。
    マージされた列の新しいエラー率（prior）は、
    「元のエラーグループから奇数個のエラーが発生する確率」として
    再計算されます。

    引数:
        dem_matrix (scipy.sparse.csc_matrix or np.ndarray): 
            圧縮前の検査行列 (H行列)。
            行が検出器、列がエラーに対応します。
            疎行列（CSC形式）を強く推奨します。
            
        priors (np.ndarray): 
            圧縮前の各エラー（列）に対応するエラー率の1Dベクトル (p)。

    戻り値:
        tuple (compressed_H, compressed_priors):
            - compressed_H (scipy.sparse.csc_matrix): 圧縮後のH行列。
            - compressed_priors (np.ndarray): 圧縮後のエラー率ベクトル (p')。
    """
    
    # 最高のパフォーマンスを得るためにCSC形式（Compressed Sparse Column）に変換
    if isspmatrix(dem_matrix) and not isinstance(dem_matrix, csc_matrix):
        H = dem_matrix.tocsc()
    elif isinstance(dem_matrix, np.ndarray):
        # 密行列は非常に遅く、メモリを大量消費する可能性があるため疎行列に変換
        H = csc_matrix(dem_matrix)
    else:
        H = dem_matrix

    if not isinstance(priors, np.ndarray):
        priors = np.array(priors)

    num_errors = H.shape[1]
    if num_errors != len(priors):
        raise ValueError(
            f"検査行列の列数 ({num_errors}) と "
            f"エラー率ベクトルの長さ ({len(priors)}) が一致しません。"
        )


    # {列ハッシュ: {"rep_index": 代表列のインデックス, "indices": [同一列のインデックスリスト]}}
    unique_cols = {}

    # 1. すべての列をスキャンし、同一の列をグループ化する
    for j in range(num_errors):
        # CSC形式から列jの非ゼロデータを効率的に取得
        # col.indices が行インデックス、col.data がその値
        col = H.getcol(j)
        
        # 列の非ゼロパターン（インデックスとデータ）を
        # 辞書のキーとして使用できる不変のタプルに変換
        # これが列の「ハッシュ」として機能する
        col_key = (tuple(col.indices), tuple(col.data))

        if col_key not in unique_cols:
            # このパターン（列）が初めて見つかった場合
            unique_cols[col_key] = {
                "rep_index": j,  # 最初の列を「代表」とする
                "indices": []
            }
        
        # このパターンを持つ列のインデックスリストに追加
        unique_cols[col_key]["indices"].append(j)


    # 2. 圧縮後のH行列とエラー率ベクトルを構築する
    new_priors_list = []
    
    # 保持する代表列のインデックスリスト
    # （元の行列からこれらの列だけを抽出するために使用）
    representative_indices = [data["rep_index"] for data in unique_cols.values()]

    for col_key, data in unique_cols.items():
        # このグループに属するすべてのエラーのインデックスを取得
        group_indices = data["indices"]
        
        # 対応するエラー率（p）の配列を取得
        group_priors = priors[group_indices]

        # 確率の計算: p' = (1 - product(1 - 2*p_k)) / 2
        # np.prodによる積の計算
        product_term = np.prod(1.0 - 2.0 * group_priors)
        
        # 新しいエラー率 p' を計算
        new_p = (1.0 - product_term) / 2.0
        
        new_priors_list.append(new_p)

    # 3. 新しい行列とベクトルを返す
    
    # 元のH行列から、保持すると決めた「代表列」だけを抽出する
    compressed_H = H[:, representative_indices]
    
    # 計算した新しいエラー率をNumPy配列に変換
    compressed_priors = np.array(new_priors_list)

    return compressed_H, compressed_priors

def get_compressed_dem_size(circuit_path: pathlib.Path) -> DemSizeInfo:
    "Read a stim circuit, compress its DEM matrix, and extract size information"

    if not circuit_path.is_file():
        raise FileNotFoundError(f"Circuit file not found: {circuit_path}")
    
    try:
        circuit = stim.Circuit.from_file(circuit_path)
        X_filtered_circuit = filter_detectors_by_basis(circuit, basis='X')
        dem = X_filtered_circuit.detector_error_model()
        matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        dem_matrix = matrices.check_matrix
        priors = matrices.priors 

        compressed_H, compressed_priors = compress_dem(dem_matrix, priors)

        num_detectors = compressed_H.shape[0]
        num_observables = dem.num_observables  
        num_errors = compressed_H.shape[1]

        return DemSizeInfo(
            filepath=circuit_path,
            num_detectors=num_detectors,
            num_observables=num_observables,
            num_errors=num_errors,
            num_instructions=len(circuit)
        )
    except Exception as e:
        raise ValueError(f"Failed to analyze circuit {circuit_path}: {e}") from e



def main():
    """
    Main function to parse arguments and analyze circuits in a directory.
    """

    parser = argparse.ArgumentParser(
        description="Analyze .stim circuit files in a directory to extract DEM size information.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--circuits_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the .stim circuit files to analyze."
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.stim",
        help="Glob pattern to match circuit files in the specified directory."
    )
    args = parser.parse_args()

    if not args.circuits_dir.is_dir():
        print(f"Error; Directory not found {args.circuit_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing circuits in: {args.circuits_dir.resolve()}")
    print(f"Matching pattern: {args.glob}\n")

    circuit_files = sorted(list(args.circuits_dir.glob(args.glob)))
    if not circuit_files:
        print("No matching circuit files found.", file=sys.stderr)
        sys.exit(0)

    all_results: List[DemSizeInfo] = []
    for circuit_path in circuit_files:
        try:
            result = analyze_stim_circuit(circuit_path)
            all_results.append(result)
        except (ValueError, FileNotFoundError) as e:
            print(e, file=sys.stderr)

    if not all_results:
        print("No circuits were successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    
    # --- 結果を整形して表示 ---
    # ヘッダーの各要素の幅を計算
    w_name = max(len(r.filepath.name) for r in all_results)
    w_det = len("Detectors")
    w_obs = len("Observables")
    w_err = len("Errors")
    w_inst = len("Instructions")

    # ヘッダーを表示
    header = (
        f"{'Filename':<{w_name}} | {'Detectors':>{w_det}} | {'Observables':>{w_obs}} | "
        f"{'Errors':>{w_err}} | {'Instructions':>{w_inst}}"
    )
    print(header)
    print("-" * len(header))

    # 各結果を表示
    for r in all_results:
        print(
            f"{r.filepath.name:<{w_name}} | {r.num_detectors:>{w_det}} | {r.num_observables:>{w_obs}} | "
            f"{r.num_errors:>{w_err}} | {r.num_instructions:>{w_inst}}"
        )


    # compressed DEM analysis
    # print("\nCompressed DEM Analysis:\n")
    # all_results: List[DemSizeInfo] = []
    # for circuit_path in circuit_files:
    #     try:
    #         result = get_compressed_dem_size(circuit_path)
    #         all_results.append(result)
    #     except (ValueError, FileNotFoundError) as e:
    #         print(e, file=sys.stderr)

    # if not all_results:
    #     print("No circuits were successfully analyzed.", file=sys.stderr)
    #     sys.exit(1)

    
    # # --- 結果を整形して表示 ---
    # # ヘッダーの各要素の幅を計算
    # w_name = max(len(r.filepath.name) for r in all_results)
    # w_det = len("Detectors")
    # w_obs = len("Observables")
    # w_err = len("Errors")
    # w_inst = len("Instructions")

    # # ヘッダーを表示
    # header = (
    #     f"{'Filename':<{w_name}} | {'Detectors':>{w_det}} | {'Observables':>{w_obs}} | "
    #     f"{'Errors':>{w_err}} | {'Instructions':>{w_inst}}"
    # )
    # print(header)
    # print("-" * len(header))

    # # 各結果を表示
    # for r in all_results:
    #     print(
    #         f"{r.filepath.name:<{w_name}} | {r.num_detectors:>{w_det}} | {r.num_observables:>{w_obs}} | "
    #         f"{r.num_errors:>{w_err}} | {r.num_instructions:>{w_inst}}"
    #     )

if __name__ == '__main__':
    main()