//
//  matrixPensilMethod.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import Numerics
import Dispatch
import SebbuScience
import NumericsExtensions
import SebbuCollections

/// Matrix pencil method routine to find exponential series representation of bath correlation functions.
public enum MatrixPencil {
    /// Fits the requested amount of terms to the given real function such that y(x) \approx \sum_i G_i e^{-W_i x}
    /// - Parameters:
    ///   - y: The samples of the function
    ///   - dt: Spacing of the evenly spaces samples
    ///   - terms: The number of exponentials to use in the exponential series fitting
    /// - Returns: Array of (G, W) pairs, where G are the coefficients/amplitudes of the exponential series and W are the exponents
    @inlinable
    public static func fit(y: [Complex<Double>], dt: Double, pencilParameter: Int? = nil, terms: Int? = nil) -> (G: [Complex<Double>], W: [Complex<Double>]) {
        let M = y.count
        let pencilParameter = pencilParameter ?? M / 2
        // Build Hankel matrices
        let Y0: Matrix<Complex<Double>> = .hankel(firstRow: .init(y[0..<pencilParameter]), lastColumn: .init(y[(pencilParameter-1)..<(M-1)]))
        let Y1: Matrix<Complex<Double>> = .hankel(firstRow: .init(y[1...pencilParameter]), lastColumn: .init(y[pencilParameter..<M]))
        // Singular value decomposition
        let (U, singularValues, VH) = try! MatrixOperations.singularValueDecomposition(A: Y0)

        let terms = terms ?? {
            let tolerance = singularValues.max()! * 1e-12
            return singularValues.count { $0 > tolerance }
        }()
        // Use "terms" most significant singular values
        let Sk: Matrix<Complex<Double>> = .diagonal(from: singularValues[0..<terms].map { Complex($0) })
        var Uk: Matrix<Complex<Double>> = .zeros(rows: U.rows, columns: terms)
        var Vk: Matrix<Complex<Double>> = .zeros(rows: terms, columns: VH.columns)
        for row in 0..<Uk.rows {
            for column in 0..<Uk.columns {
                Uk[row, column] = U[row, column]
            }
        }
        for row in 0..<Vk.rows {
            for column in 0..<Vk.columns {
                Vk[row, column] = VH[row, column]
            }
        }
        Vk = Vk.conjugateTranspose
        Uk = Uk.conjugateTranspose

        // Reduced pencil matrix to obtain eigenvalues -> exponents W
        let _V = Uk.dot(Y1).dot(Vk)
        let A = try! MatrixOperations.solve(A: Sk, B: _V)

        let eigenvalues = try! MatrixOperations.eigenValues(A)
        let W = eigenvalues.map {-Complex.log($0) / dt}

        // Solve amplitudes G by least squares
        var vandermond: Matrix<Complex<Double>> = .zeros(rows: M, columns: W.count)
        for n in 0..<vandermond.rows {
            for m in 0..<vandermond.columns {
                vandermond[n, m] = .exp(-Double(n) * dt * W[m])
            }
        }
        let yVec = Vector(y)
        let (G, _) = try! Optimize.linearLeastSquares(A: vandermond, yVec)
        return (G.components, W)
    }
}
