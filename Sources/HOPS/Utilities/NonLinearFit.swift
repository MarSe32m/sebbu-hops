//
//  NonLinearFit.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 6.6.2026.
//

import Numerics
import SebbuScience
import NumericsExtensions

public struct PhysicalBCFInitializer {
    public static func initialRFromLiftedQ(
        G targetG: [Complex<Double>],
        W: [Complex<Double>],
        ridgeScale: Double = 1e-12,
        diagonalLoadEpsilon: Double = 1e-13,
        powerIterations: Int = 200
    ) -> (r: [Complex<Double>], Q: Matrix<Complex<Double>>, relativeResidual: Double) {
        precondition(targetG.count == W.count)

        let N = W.count
        let variableCount = N * N

        // We solve:
        //
        //     G_j ≈ sum_k Q[j,k] / (W[j] + conj(W[k]))
        //
        // with Q Hermitian.
        //
        // Hermitian Q has N^2 real variables:
        //
        //     Q[j,j] real
        //     Q[j,k] = a_jk + i b_jk, j < k
        //
        // Unknown ordering:
        //
        //     0..<N: diagonal variables Q[j,j]
        //     then Re Q[j,k], Im Q[j,k] for j < k

        var realIndex = Array(repeating: Array(repeating: -1, count: N), count: N)
        var imagIndex = Array(repeating: Array(repeating: -1, count: N), count: N)

        var column = N
        for j in 0..<N {
            for k in (j + 1)..<N {
                realIndex[j][k] = column
                column += 1

                imagIndex[j][k] = column
                column += 1
            }
        }

        precondition(column == variableCount)

        // Main equations: 2N real equations.
        // Ridge equations: variableCount equations.
        let mainRows = 2 * N
        let totalRows = mainRows + variableCount

        var A: Matrix<Double> = .zeros(rows: totalRows, columns: variableCount)
        var b = Vector<Double>(Array(repeating: 0.0, count: totalRows))

        var frobeniusSquared = 0.0

        func addCoefficient(_ value: Double, row: Int, column: Int) {
            A[row, column] += value
            frobeniusSquared += value * value
        }

        for j in 0..<N {
            let realRow = 2 * j
            let imagRow = 2 * j + 1

            b[realRow] = targetG[j].real
            b[imagRow] = targetG[j].imaginary

            // Diagonal variable Q[j,j]
            do {
                let coeff = Complex<Double>(1.0, 0.0) / (W[j] + W[j].conjugate)
                let col = j

                addCoefficient(coeff.real, row: realRow, column: col)
                addCoefficient(coeff.imaginary, row: imagRow, column: col)
            }

            // Variables Q[p,q] for p < q.
            // Only variables touching row j contribute to G_j.
            for k in 0..<N where k != j {
                if j < k {
                    // Q[j,k] = a + i b
                    let coeff = Complex<Double>(1.0, 0.0) / (W[j] + W[k].conjugate)

                    let aCol = realIndex[j][k]
                    let bCol = imagIndex[j][k]

                    // a contribution: coeff * a
                    addCoefficient(coeff.real, row: realRow, column: aCol)
                    addCoefficient(coeff.imaginary, row: imagRow, column: aCol)

                    // b contribution: coeff * i b
                    let iCoeff = Complex<Double>(-coeff.imaginary, coeff.real)

                    addCoefficient(iCoeff.real, row: realRow, column: bCol)
                    addCoefficient(iCoeff.imaginary, row: imagRow, column: bCol)
                } else {
                    // j > k:
                    //
                    // Q[j,k] = conj(Q[k,j]) = a - i b
                    let coeff = Complex<Double>(1.0, 0.0) / (W[j] + W[k].conjugate)

                    let aCol = realIndex[k][j]
                    let bCol = imagIndex[k][j]

                    // a contribution: coeff * a
                    addCoefficient(coeff.real, row: realRow, column: aCol)
                    addCoefficient(coeff.imaginary, row: imagRow, column: aCol)

                    // b contribution: coeff * (-i) b
                    let minusICoeff = Complex<Double>(coeff.imaginary, -coeff.real)

                    addCoefficient(minusICoeff.real, row: realRow, column: bCol)
                    addCoefficient(minusICoeff.imaginary, row: imagRow, column: bCol)
                }
            }
        }

        // Ridge regularization:
        //
        //     min ||Aq - b||^2 + eta ||q||^2
        //
        // implemented by augmenting A with sqrt(eta) I.
        let eta = ridgeScale * max(frobeniusSquared, 1.0)
        let sqrtEta = eta.squareRoot()

        for col in 0..<variableCount {
            let row = mainRows + col
            A[row, col] = sqrtEta
            b[row] = 0.0
        }

        let (q, _) = try! Optimize.linearLeastSquares(A: A, b)

        // Reconstruct Hermitian Q.
        var Q: Matrix<Complex<Double>> = .zeros(rows: N, columns: N)

        for j in 0..<N {
            Q[j, j] = Complex<Double>(q[j], 0.0)
        }

        for j in 0..<N {
            for k in (j + 1)..<N {
                let a = q[realIndex[j][k]]
                let im = q[imagIndex[j][k]]
                let value = Complex<Double>(a, im)

                Q[j, k] = value
                Q[k, j] = value.conjugate
            }
        }

        Q = symmetrizedHermitian(Q)

        // Cheap PSD enforcement.
        // This is conservative but robust and avoids requiring a Hermitian eigensolver.
        Q = diagonallyLoadedPSD(Q, epsilon: diagonalLoadEpsilon)

        // Extract approximate rank-one factor Q ≈ r r†.
        let leading = leadingEigenpairHermitianPower(
            Q,
            maxIterations: powerIterations
        )

        var r = Array(repeating: Complex<Double>.zero, count: N)

        let lambda = max(leading.value, 0.0)
        let amplitude = lambda.squareRoot()

        for j in 0..<N {
            r[j] = leading.vector[j] * Complex<Double>(amplitude, 0.0)
        }

        // Rescale r so that G(r) best matches targetG in a real least-squares sense.
        r = rescaledR(r, W: W, targetG: targetG)

        // Fix global phase gauge: largest component becomes real and positive.
        r = phaseGaugeFixed(r)

        let fittedG = amplitudesFromR(r, W: W)
        let relativeResidual = relativeComplexResidual(fittedG, targetG)

        return (r, Q, relativeResidual)
    }

    public static func amplitudesFromR(
        _ r: [Complex<Double>],
        W: [Complex<Double>]
    ) -> [Complex<Double>] {
        let N = W.count
        precondition(r.count == N)

        var G = Array(repeating: Complex<Double>.zero, count: N)

        for j in 0..<N {
            var sum = Complex<Double>.zero

            for k in 0..<N {
                sum += r[j] * r[k].conjugate / (W[j] + W[k].conjugate)
            }

            G[j] = sum
        }

        return G
    }

    private static func rescaledR(
        _ r: [Complex<Double>],
        W: [Complex<Double>],
        targetG: [Complex<Double>]
    ) -> [Complex<Double>] {
        let currentG = amplitudesFromR(r, W: W)

        var numerator = 0.0
        var denominator = 0.0

        for j in 0..<targetG.count {
            numerator += (currentG[j].conjugate * targetG[j]).real
            denominator += currentG[j].lengthSquared
        }

        guard denominator > 0 else {
            return r
        }

        let scale = max(numerator / denominator, 0.0)
        let factor = scale.squareRoot()

        return r.map { $0 * Complex<Double>(factor, 0.0) }
    }

    private static func phaseGaugeFixed(
        _ r: [Complex<Double>]
    ) -> [Complex<Double>] {
        guard !r.isEmpty else { return r }

        var pivot = 0
        var pivotAbs2 = r[0].lengthSquared

        for j in 1..<r.count {
            let value = r[j].lengthSquared
            if value > pivotAbs2 {
                pivot = j
                pivotAbs2 = value
            }
        }

        guard pivotAbs2 > 0 else {
            return r
        }
        
        let phase = Double.atan2(y: r[pivot].imaginary, x: r[pivot].real)
        let gauge = Complex<Double>(.cos(phase), -.sin(phase))

        return r.map { $0 * gauge }
    }

    private static func symmetrizedHermitian(
        _ Q: Matrix<Complex<Double>>
    ) -> Matrix<Complex<Double>> {
        var result = Q

        for j in 0..<Q.rows {
            for k in 0..<Q.columns {
                result[j, k] = Complex<Double>(0.5, 0.0) * (Q[j, k] + Q[k, j].conjugate)
            }
        }

        for j in 0..<Q.rows {
            result[j, j] = Complex<Double>(result[j, j].real, 0.0)
        }

        return result
    }

    private static func diagonallyLoadedPSD(
        _ Q: Matrix<Complex<Double>>,
        epsilon: Double
    ) -> Matrix<Complex<Double>> {
        let N = Q.rows
        var minGershgorinLowerBound = Double.infinity

        for j in 0..<N {
            var radius = 0.0

            for k in 0..<N where k != j {
                radius += Q[j, k].length
            }

            let lowerBound = Q[j, j].real - radius
            minGershgorinLowerBound = min(minGershgorinLowerBound, lowerBound)
        }

        let shift = max(0.0, epsilon - minGershgorinLowerBound)

        var result = Q
        if shift > 0 {
            for j in 0..<N {
                result[j, j] += Complex<Double>(shift, 0.0)
            }
        }

        return result
    }

    private static func leadingEigenpairHermitianPower(
        _ A: Matrix<Complex<Double>>,
        maxIterations: Int
    ) -> (value: Double, vector: [Complex<Double>]) {
        let N = A.rows
        precondition(A.columns == N)

        var bestValue = -Double.infinity
        var bestVector = Array(repeating: Complex<Double>.zero, count: N)

        // Deterministic starts: all-ones plus each coordinate basis vector.
        var starts: [[Complex<Double>]] = []

        let invSqrtN = 1.0 / Double(max(N, 1)).squareRoot()
        starts.append(Array(repeating: Complex<Double>(invSqrtN, 0.0), count: N))

        for j in 0..<N {
            var e = Array(repeating: Complex<Double>.zero, count: N)
            e[j] = Complex<Double>(1.0, 0.0)
            starts.append(e)
        }

        for start in starts {
            var x = normalized(start)

            for _ in 0..<maxIterations {
                let y = matVec(A, x)
                let normY = (y.reduce(0.0) { $0 + $1.lengthSquared }).squareRoot()

                if normY == 0 {
                    break
                }

                x = y.map { $0 / Complex<Double>(normY, 0.0) }
            }

            let Ax = matVec(A, x)

            var rayleigh = Complex<Double>.zero
            for j in 0..<N {
                rayleigh += x[j].conjugate * Ax[j]
            }

            let value = rayleigh.real

            if value > bestValue {
                bestValue = value
                bestVector = x
            }
        }

        return (bestValue, bestVector)
    }

    private static func matVec(
        _ A: Matrix<Complex<Double>>,
        _ x: [Complex<Double>]
    ) -> [Complex<Double>] {
        let N = A.rows
        precondition(A.columns == x.count)

        var y = Array(repeating: Complex<Double>.zero, count: N)

        for i in 0..<N {
            var sum = Complex<Double>.zero

            for j in 0..<x.count {
                sum += A[i, j] * x[j]
            }

            y[i] = sum
        }

        return y
    }

    private static func normalized(
        _ x: [Complex<Double>]
    ) -> [Complex<Double>] {
        let norm = (x.reduce(0.0) { $0 + $1.lengthSquared }).squareRoot()

        guard norm > 0 else {
            return x
        }

        return x.map { $0 / Complex<Double>(norm, 0.0) }
    }

    private static func relativeComplexResidual(
        _ fitted: [Complex<Double>],
        _ target: [Complex<Double>]
    ) -> Double {
        precondition(fitted.count == target.count)

        var numerator = 0.0
        var denominator = 0.0

        for j in 0..<fitted.count {
            numerator += (fitted[j] - target[j]).lengthSquared
            denominator += (target[j]).lengthSquared
        }

        return (numerator / max(denominator, 1e-300)).squareRoot()
    }
}

@usableFromInline
struct UnconstrainedBCFFitProblem {
    @usableFromInline
    internal let N: Int
    @usableFromInline
    internal let gammaMin: Double
    @usableFromInline
    internal let tau: [Double]
    @usableFromInline
    internal let bcfTarget: [Complex<Double>]
    @usableFromInline
    internal let weights: [Double]
    
    @inlinable
    func unpack(_ theta: Vector<Double>) -> (G: Vector<Complex<Double>>, W: Vector<Complex<Double>>) {
        precondition(theta.count == 4 * N)
        var G: Vector<Complex<Double>> = .zero(N)
        var W: Vector<Complex<Double>> = .zero(N)
        
        for j in 0..<N {
            let a = theta[j]
            let omega = theta[N + j]
            let x = theta[2 * N + j]
            let y = theta[3 * N + j]
            W[j] = Complex(gammaMin + .exp(a), omega)
            G[j] = Complex(x, y)
        }
        return (G, W)
    }
    
    @inlinable
    func bcfFit(theta: Vector<Double>) -> Vector<Complex<Double>> {
        let (G, W) = unpack(theta)
        var result: Vector<Complex<Double>> = .zero(tau.count)
        for n in 0..<tau.count {
            result[n] = .zero
            for j in 0..<N {
                result[n] += G[j] * .exp(-tau[n] * W[j])
            }
        }
        return result
    }
    
    @inlinable
    func residuals(theta: Vector<Double>) -> Vector<Double> {
        let bcf = bcfFit(theta: theta)
        var result: Vector<Double> = .zero(2 * tau.count)
        for n in 0..<tau.count {
            let diff = bcf[n] - bcfTarget[n]
            let sqrtWeight: Double = weights[n].squareRoot()
            result[n] = sqrtWeight * diff.real.magnitude
            result[tau.count + n] = sqrtWeight * diff.imaginary.magnitude
            
        }
        return result
    }
}

@usableFromInline
struct PhysicalBCFFitProblem {
    @usableFromInline
    internal let N: Int
    @usableFromInline
    internal let gammaMin: Double
    @usableFromInline
    internal let tau: [Double]
    @usableFromInline
    internal let bcfTarget: [Complex<Double>]
    @usableFromInline
    internal let weights: [Double]
    
    @inlinable
    func unpack(_ theta: Vector<Double>) -> (r: Vector<Complex<Double>>, lambda: Vector<Complex<Double>>) {
        precondition(theta.count == 4 * N)
        var r: Vector<Complex<Double>> = .zero(N)
        var lambda: Vector<Complex<Double>> = .zero(N)
        
        for j in 0..<N {
            let a = theta[j]
            let omega = theta[N + j]
            let x = theta[2 * N + j]
            let y = theta[3 * N + j]
            lambda[j] = Complex(gammaMin + .exp(a), omega)
            r[j] = Complex(x, y)
        }
        return (r, lambda)
    }
    
    @inlinable
    func coefficients(r: Vector<Complex<Double>>, lambda: Vector<Complex<Double>>) -> Vector<Complex<Double>> {
        var G: Vector<Complex<Double>> = .zero(N)
        for j in 0..<N {
            var sum: Complex<Double> = .zero
            for k in 0..<N {
                let numerator = r[j] * r[k].conjugate
                let denominator = lambda[j] + lambda[k].conjugate
                sum += numerator / denominator
            }
            G[j] = sum
        }
        return G
    }
    
    @inlinable
    func bcfFit(theta: Vector<Double>) -> Vector<Complex<Double>> {
        let (r, lambda) = unpack(theta)
        let G = coefficients(r: r, lambda: lambda)
        var result: Vector<Complex<Double>> = .zero(tau.count)
        for n in 0..<tau.count {
            result[n] = .zero
            for j in 0..<N {
                result[n] += G[j] * .exp(-tau[n] * lambda[j])
            }
        }
        return result
    }
    
    @inlinable
    func residuals(theta: Vector<Double>) -> Vector<Double> {
        let bcf = bcfFit(theta: theta)
        var result: Vector<Double> = .zero(2 * tau.count)
        for n in 0..<tau.count {
            let diff = bcf[n] - bcfTarget[n]
            let sqrtWeight: Double = weights[n].squareRoot()
            result[n] = sqrtWeight * diff.real.magnitude
            result[tau.count + n] = sqrtWeight * diff.imaginary.magnitude
            
        }
        return result
    }
}

struct LeastSquaresResult {
    var theta: Vector<Double>
    var cost: Double
    var iterations: Int
    var reason: String
}

func inifinityNorm(_ x: Vector<Double>) -> Double {
    var max = 0.0
    for x in x.components {
        if x.magnitude > max { max = x.magnitude }
    }
    return max
}

func finiteDifferenceJacobian(
    theta: Vector<Double>,
    residual: (Vector<Double>) -> Vector<Double>,
    eps: Double = 1e-6
) -> Matrix<Double> {
    let r0 = residual(theta)
    let m = r0.count
    let p = theta.count
    var thetaPlus = theta
    var thetaMinus = theta
    var J: Matrix<Double> = .zeros(rows: m, columns: p)
    for j in 0..<p {
        let h = eps * (1.0 + theta[j].magnitude)
        thetaPlus[j] += h
        thetaMinus[j] -= h
        let rPlus = residual(thetaPlus)
        let rMinus = residual(thetaMinus)
        for i in 0..<m {
            J[i, j] = (rPlus[i] - rMinus[i]) / (2 * h)
        }
        thetaPlus[j] = theta[j]
        thetaMinus[j] = theta[j]
    }
    return J
}

func leastSquares(
    x0: Vector<Double>,
    maxIterations: Int = 200,
    ftol: Double = 1e-12,
    xtol: Double = 1e-12,
    gtol: Double = 1e-12,
    residual: (Vector<Double>) -> Vector<Double>
) -> Vector<Double>? {
//    Optimize.gaussNewton(initial: x0, maxIterations: maxIterations, residuals: residual) { parameters in
//        finiteDifferenceJacobian(theta: parameters, residual: residual)
//    }
    let result = Optimize.levenbergMarquardt(initial: x0, maxIterations: maxIterations, stepTolerance: xtol, costTolerance: ftol, gradientTolerance: gtol, residuals: residual)
    if !result.converged { return nil }
    return result.parameters
}


/// Matrix pencil method routine to find exponential series representation of bath correlation functions.
public enum NonLinearFit {
    /// Fits the requested amount of terms to the given real function such that y(x) \approx \sum_i G_i e^{-W_i x}
    public static func fit(t: [Double], y: [Complex<Double>], terms: Int, initialGuess: [Complex<Double>]? = nil, weights: [Double]? = nil) -> (G: [Complex<Double>], W: [Complex<Double>]) {
        let unconstrainedProblem = UnconstrainedBCFFitProblem(N: terms, gammaMin: 1e-6, tau: t, bcfTarget: y, weights: weights ?? .init(repeating: 1.0, count: y.count))
        //TODO: Use initialGuess if supplied
        var theta0: Vector<Double> = .zero(4 * terms)
        let (G, W) = MatrixPencil.fit(y: y, dt: t[1] - t[0], terms: terms)
        // Decay initial guesses
        for j in 0..<terms {
            theta0[j] = .log(W[j].real)
        }
        // Frequence initial guesses
        for j in 0..<terms {
            // Unconstrained fit
            theta0[terms + j] = W[j].imaginary
        }
        // Residue initial guesses
        for j in 0..<terms {
            // Unconstrained fit
            theta0[2 * terms + j] = G[j].real
            theta0[3 * terms + j] = G[j].imaginary
        }
        //TODO: Keep W[j] fixed and fit only r[j]
        //TODO: Then refine with both being able to move...
        let result = leastSquares(x0: theta0, maxIterations: 50000, ftol: 1e-16, xtol: 1e-16) { theta in
            unconstrainedProblem.residuals(theta: theta)
//            physicalProblem.residuals(theta: theta)
        }
        
        guard let result else {
            return (G, W)
        }
        let (_G, _W) = unconstrainedProblem.unpack(result)
        return (_G.components, _W.components)
    }
    
    /// Fits the requested amount of terms to the given real function such that y(x) \approx \sum_i G_i e^{-W_i x}
    @inlinable
    public static func fit(t: [Double], y: [Double], terms: Int, initialGuess: [Complex<Double>]? = nil, weights: [Double]? = nil) -> (G: [Complex<Double>], W: [Complex<Double>]) {
        fit(t: t, y: y.map { Complex($0)}, terms: terms, initialGuess: initialGuess, weights: weights)
    }
    
    /// Fits the requested amount of terms to the given real function such that y(x) \approx \sum_i G_i e^{-W_i x}
    public static func fitPhysical(t: [Double], y: [Complex<Double>], terms: Int, initialGuess: [Complex<Double>]? = nil, weights: [Double]? = nil) -> (G: [Complex<Double>], W: [Complex<Double>], r: [Complex<Double>]) {
        let physicalProblem = PhysicalBCFFitProblem(N: terms, gammaMin: 1e-6, tau: t, bcfTarget: y, weights: weights ?? .init(repeating: 1.0, count: y.count))
        //TODO: Use initialGuess if supplied
        var theta0: Vector<Double> = .zero(4 * terms)
        let (G, W) = MatrixPencil.fit(y: y, dt: t[1] - t[0], terms: terms)
        let initializer = PhysicalBCFInitializer.initialRFromLiftedQ(G: G, W: W)
        let r0 = initializer.r
        // Decay initial guesses
        for j in 0..<terms {
            // Constrained fit
            theta0[j] = .log(W[j].real)
        }
        // Frequence initial guesses
        for j in 0..<terms {
            // Constrained fit
            theta0[terms + j] = W[j].imaginary
        }
        // Residue initial guesses
        for j in 0..<terms {
            // Constrained fit
            theta0[2 * terms + j] = r0[j].real
            theta0[3 * terms + j] = r0[j].imaginary
//            theta0[2 * terms + j] = 0.1 * Double.random(in: -1 ... 1)
//            theta0[3 * terms + j] = 0.1 * Double.random(in: -1 ... 1)
        }
        //TODO: Keep W[j] fixed and fit only r[j]
        //TODO: Then refine with both being able to move...
        let result = leastSquares(x0: theta0, maxIterations: 50000, ftol: 1e-16, xtol: 1e-16) { theta in
            physicalProblem.residuals(theta: theta)
        }
        guard let result else {
            return (G, W, .init(repeating: .zero, count: terms))
        }
        let (r, lambda) = physicalProblem.unpack(result)
        let _G = physicalProblem.coefficients(r: r, lambda: lambda)
        return (_G.components, lambda.components, r.components)
    }
    
    /// Fits the requested amount of terms to the given real function such that y(x) \approx \sum_i G_i e^{-W_i x}
    @inlinable
    public static func fitPhysical(t: [Double], y: [Double], terms: Int, initialGuess: [Complex<Double>]? = nil, weights: [Double]? = nil) -> (G: [Complex<Double>], W: [Complex<Double>], r: [Complex<Double>]) {
        fitPhysical(t: t, y: y.map { Complex($0)}, terms: terms, initialGuess: initialGuess, weights: weights)
    }
}
