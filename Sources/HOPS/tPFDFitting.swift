//
//  tPFDFitting.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import Numerics
import Dispatch
import SebbuScience
import NumericsExtensions
import SebbuCollections

/// tPFD fitting routine to find exponential series representation of bath correlation functions.
/// This is based on the method presented in "Universal time-domain Prony fitting decomposition for optimized hierarchical quantum master equations", J. Chem. Phys. 156, 221102 (2022): https://doi.org/10.1063/5.0095961
public enum tPFD {
    /// Fits the requested amount of terms to the given real function such that y(x) \approx \sum_i G_i e^{-W_i x}
    /// - Parameters:
    ///   - x: The x-axis of the function. There must be an odd number of elements in this array!
    ///   - y: The y-axis of the function. There must be an odd number of elements in this array!
    ///   - terms: The number of exponentials to use in the exponential series fitting
    /// - Returns: Array of (G, W) pairs, where G are the coefficients of the exponential series and W the exponents
    @inlinable
    public static func fit(x: [Double], y: [Double], terms: Int) -> (G: [Complex<Double>], W: [Complex<Double>]) {
        precondition(x.count % 2 == 1, "There needs to be an odd number of samples for the tPFD fit")
        precondition(x.count == y.count)
        // Takagi factorization
        let dt = x[1] - x[0]
        let H = Matrix<Double>.hankel(firstRow: Array(y[0...y.count/2]), lastColumn: Array(y[y.count/2..<y.count]))
        let (_, U) = try! MatrixOperations.takagiSymmetric(H)
        
        // Polynomial root finding (currently the bottleneck of this method)
        let u = U[terms]
        let polynomial = Polynomial(u)
        let z = polynomial.roots().sorted(by: { $0.length < $1.length })[0..<terms]
        let W = z.map { Complex<Double>(-1.0 / dt * Double.log($0.length), -1.0 / dt * $0.phase) }
        
        // Least squares fitting
        var A = Matrix<Complex<Double>>.zeros(rows: terms, columns: terms)
        var a = Vector<Complex<Double>>.zero(terms)
        for mu in 0..<terms {
            for nu in mu..<terms {
                let z_mu = z[mu]
                let z_nu = z[nu]
                let z_mu_z_nu = z_mu * z_nu
                var A_nu_mu: Complex<Double> = .zero
                for i in 0..<y.count {
                    A_nu_mu += Complex<Double>.pow(z_mu_z_nu, i)
                }
                A[nu, mu] = A_nu_mu
                A[mu, nu] = A_nu_mu
            }
            for i in 0..<y.count {
                a[mu] += y[i] * Complex<Double>.pow(z[mu], i)
            }
        }
        let G = try! MatrixOperations.solve(A: A, b: a)
        return (G.components, W)
    }
    
    /// Fits the requested amount of exponential terms to the given complex function such that y(x) \approx \sum_i G_i e^{-W_i x}
    /// - Parameters:
    ///   - x: The x-axis of the function. There must be an odd number of elements in this array!
    ///   - y: The y-axis of the function. There must be an odd number of elements in this array!
    ///   - realTerms: The number of exponentials to use in the exponential series fitting of the real part of the samples
    ///   - imaginaryTerms: The number of exponentals to use in the exponential series fitting of the imaginary part of the samples
    /// - Returns: Array of (G, W) pairs, where G are the coefficients of the exponential series and W the exponents
    @inlinable
    public static func fit(x: [Double], y: [Complex<Double>], realTerms: Int, imaginaryTerms: Int) -> (G: [Complex<Double>], W: [Complex<Double>]) {
        nonisolated(unsafe) var resultReal: (G: [Complex<Double>], W: [Complex<Double>]) = ([], [])
        nonisolated(unsafe) var resultImaginary: (G: [Complex<Double>], W: [Complex<Double>]) = ([], [])
        DispatchQueue.concurrentPerform(iterations: 2) { i in
            if i == 0 {
                resultReal = fit(x: x, y: y.real, terms: realTerms)
            } else {
                resultImaginary = fit(x: x, y: y.imaginary, terms: imaginaryTerms)
            }
        }
        var result = resultReal
        for (G, W) in zip(resultImaginary.G, resultImaginary.W) {
            result.G.append(.i * G)
            result.W.append(W)
        }
        return result
    }
}
