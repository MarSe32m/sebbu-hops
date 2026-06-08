//
//  matrixPensilMethod.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import Numerics
import SebbuScience
import NumericsExtensions

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
        //let A = try! MatrixOperations.solve(A: Sk, B: _V)
        let A = Matrix<Complex<Double>>(rows: _V.rows, columns: _V.columns) { buffer in
            var index = 0
            for i in 0..<_V.rows {
                for j in 0..<_V.columns {
                    buffer[index] = _V[i, j] / Sk[i, i]
                    index += 1
                }
            }
        }

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
    
    /// Fits the requested amount of terms to the given real function such that y(x) \approx \sum_i G_i e^{-W_i x}
    /// - Parameters:
    ///   - y: The samples of the function
    ///   - dt: Spacing of the evenly spaces samples
    ///   - terms: The number of exponentials to use in the exponential series fitting
    /// - Returns: Array of (G, W) pairs, where G are the coefficients/amplitudes of the exponential series and W are the exponents
    @inlinable
    public static func fit(y: [Double], dt: Double, pencilParameter: Int? = nil, terms: Int? = nil) -> (G: [Complex<Double>], W: [Complex<Double>]) {
        fit(y: y.map { Complex($0) }, dt: dt, pencilParameter: pencilParameter, terms: terms)
    }
    
    public static func fit2(y: [Double], dt: Double, pencilParameter: Int? = nil, terms: Int? = nil) -> (G: [Complex<Double>], W: [Complex<Double>]) {
        fit2(y: y.map { Complex($0) }, dt: dt, pencilParameter: pencilParameter, terms: terms)
    }
    
    public static func fit2(
        y: [Complex<Double>],
        dt: Double,
        pencilParameter: Int? = nil,
        terms: Int? = nil
    ) -> (G: [Complex<Double>], W: [Complex<Double>]) {
        let M = y.count
        let pencilParameter = pencilParameter ?? M / 2

        // Build Hankel matrices
        let Y0: Matrix<Complex<Double>> = .hankel(
            firstRow: .init(y[0..<pencilParameter]),
            lastColumn: .init(y[(pencilParameter - 1)..<(M - 1)])
        )

        let Y1: Matrix<Complex<Double>> = .hankel(
            firstRow: .init(y[1...pencilParameter]),
            lastColumn: .init(y[pencilParameter..<M])
        )

        // Singular value decomposition
        let (U, singularValues, VH) = try! MatrixOperations.singularValueDecomposition(A: Y0)

        let terms = terms ?? {
            let tolerance = singularValues.max()! * 1e-12
            return singularValues.count { $0 > tolerance }
        }()

        // Use "terms" most significant singular values
        let Sk: Matrix<Complex<Double>> = .diagonal(
            from: singularValues[0..<terms].map { Complex($0) }
        )

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

        let A = Matrix<Complex<Double>>(rows: _V.rows, columns: _V.columns) { buffer in
            var index = 0
            for i in 0..<_V.rows {
                for j in 0..<_V.columns {
                    buffer[index] = _V[i, j] / Sk[i, i]
                    index += 1
                }
            }
        }

        let eigenvalues = try! MatrixOperations.eigenValues(A)
        let W = eigenvalues.map { -Complex.log($0) / dt }

        // ------------------------------------------------------------------
        // Constrained amplitude fit:
        //
        //     G_j = u_j + i v_j
        //
        // with constraint:
        //
        //     sum_j v_j = 0
        //
        // We eliminate the final imaginary component:
        //
        //     v_{N-1} = -sum_{j=0}^{N-2} v_j
        //
        // Unknown vector:
        //
        //     theta = [u_0, ..., u_{N-1}, v_0, ..., v_{N-2}]
        //
        // Number of unknowns = 2N - 1.
        // ------------------------------------------------------------------

        let N = W.count
        let numberOfUnknowns = 2 * N - 1
        let numberOfRows = 2 * M

        var design: Matrix<Double> = .zeros(rows: numberOfRows, columns: numberOfUnknowns)
        var rhs: Vector<Double> = .zero(numberOfRows)

        for n in 0..<M {
            let t = Double(n) * dt

            var basis = Array(repeating: Complex<Double>.zero, count: N)
            for j in 0..<N {
                basis[j] = .exp(-t * W[j])
            }

            let lastBasis = basis[N - 1]

            let realRow = 2 * n
            let imagRow = 2 * n + 1

            // Columns for real parts u_j
            for j in 0..<N {
                let coeff = basis[j]

                design[realRow, j] = coeff.real
                design[imagRow, j] = coeff.imaginary
            }

            // Columns for imaginary parts v_j, j = 0..<N-1
            //
            // Contribution from v_j is:
            //
            //     i v_j (basis[j] - basis[N - 1])
            //
            // because v_{N-1} has been eliminated.
            for j in 0..<(N - 1) {
                let coeff = Complex<Double>.i * (basis[j] - lastBasis)
                let column = N + j

                design[realRow, column] = coeff.real
                design[imagRow, column] = coeff.imaginary
            }

            rhs[realRow] = y[n].real
            rhs[imagRow] = y[n].imaginary
        }

        let (theta, _) = try! Optimize.linearLeastSquares(A: design, rhs)

        var G = Array(repeating: Complex<Double>.zero, count: N)

        // Real parts u_j
        for j in 0..<N {
            G[j].real = theta[j]
        }

        // Imaginary parts v_j for j = 0..<N-1
        var sumImaginary = 0.0
        for j in 0..<(N - 1) {
            let vj = theta[N + j]
            G[j].imaginary = vj
            sumImaginary += vj
        }

        // Enforce the constraint exactly
        G[N - 1].imaginary = -sumImaginary

        return (G, W)
    }
}
