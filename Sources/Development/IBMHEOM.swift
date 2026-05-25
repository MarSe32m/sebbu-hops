//
//  IBM.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import HEOM
import HOPS
import SebbuScience
import PythonKitUtilities
import Synchronization

public func IBMExampleHEOM(endTime: Double = 7.0, depth: Int, plotBCF: Bool = false) -> (tSpace: [Double], X: [Double], Y: [Double], Z: [Double]) {
    let A = 0.027
    let omegaC = 1.447
    
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, omegaC: omegaC)
    }
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        //let (G, W) = tPFD.fit(x: tSpace, y: bcf, realTerms: 4, imaginaryTerms: 4)
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        if plotBCF {
            let tSpace = [Double].linearSpace(0, 20, 501)
            let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
            let bcfExponentials = tSpace.map { t in
                var result: Complex<Double> = .zero
                for i in 0..<G.count {
                    result += G[i] * .exp(-t * W[i])
                }
                return result
            }
            plt.figure()
            plt.plot(x: tSpace, y: bcf.real, label: "Real")
            plt.plot(x: tSpace, y: bcf.imaginary, label: "Imaginary")
            plt.plot(x: tSpace, y: bcfExponentials.real, label: "Real", linestyle: "--")
            plt.plot(x: tSpace, y: bcfExponentials.imaginary, label: "Imaginary", linestyle: "--")
            plt.legend()
            plt.show()
            plt.close()
        }
        return (G, W)
    }()
    
    let sigmaMinus: Matrix<Complex<Double>> = .init(elements: [
        .zero, .one,
        .zero, .zero
    ], rows: 2, columns: 2)
    let rate = 0.0175 * 0
    
    let _initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    let initialState = _initialState.outer(_initialState.conjugate)
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    
    let start = ContinuousClock.now
    let hierarchy = HEOMHierarchy2(dimension: 2, L: L, G: G, W: W, depth: depth)
    let (tSpace, rho) = hierarchy.solve(end: endTime, initialState: initialState, H: H, lindbladians: [(rate, sigmaMinus)], stepSize: 0.1)
    let end = ContinuousClock.now
    print("HEOM: \(end - start)")
    
    let X = rho.map { 2 * $0[0, 1].real }
    let Y = rho.map { 2 * $0[0, 1].imaginary }
    let Z = rho.map { $0[0, 0].real - $0[1, 1].real }
    return (tSpace, X, Y, Z)
}

public func _linearIBM(endTime: Double = 7.0, realizations: Int, plotBCF: Bool = false) -> (tSpace: [Double], X: [Double], Y: [Double], Z: [Double]) {
    let A = 0.027
    let omegaC = 1.447
    
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, omegaC: omegaC)
    }
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        //let (G, W) = tPFD.fit(x: tSpace, y: bcf, realTerms: 4, imaginaryTerms: 4)
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        if plotBCF {
            let tSpace = [Double].linearSpace(0, 20, 501)
            let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
            let bcfExponentials = tSpace.map { t in
                var result: Complex<Double> = .zero
                for i in 0..<G.count {
                    result += G[i] * .exp(-t * W[i])
                }
                return result
            }
            plt.figure()
            plt.plot(x: tSpace, y: bcf.real, label: "Real")
            plt.plot(x: tSpace, y: bcf.imaginary, label: "Imaginary")
            plt.plot(x: tSpace, y: bcfExponentials.real, label: "Real", linestyle: "--")
            plt.plot(x: tSpace, y: bcfExponentials.imaginary, label: "Imaginary", linestyle: "--")
            plt.legend()
            plt.show()
            plt.close()
        }
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 5)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime, dtMax: 0.01) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    nonisolated(unsafe) var tSpace: [Double] = []
    nonisolated(unsafe) var X: [Double] = []
    nonisolated(unsafe) var Y: [Double] = []
    nonisolated(unsafe) var Z: [Double] = []
    
    let mutex: Mutex<Void> = Mutex(())
    
    let linearStart = ContinuousClock().now
    (0..<realizations).parallelForEach { index in
        defer {
            if (index + 1) % 100 == 0 {
                print("Trajectory \(index + 1)")
            }
        }
        let noise = zGenerator.generate()
        let (_tSpace, trajectory) = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: noise, stepSize: 0.01)
        let rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory)
        let x = rho.map { 2 * $0[0, 1].real }
        let y = rho.map { 2 * $0[0, 1].imaginary }
        let z = rho.map { $0[0, 0].real - $0[1, 1].real }
        
        mutex.withLock { _ in
            if tSpace.isEmpty {
                tSpace = _tSpace
                X = x.map { $0 / Double(realizations) }
                Y = y.map { $0 / Double(realizations) }
                Z = z.map { $0 / Double(realizations) }
            } else {
                for i in 0..<X.count {
                    X[i] += x[i] / Double(realizations)
                    Y[i] += y[i] / Double(realizations)
                    Z[i] += z[i] / Double(realizations)
                }
            }
        }
    }
    let linearEnd = ContinuousClock().now
    print("Linear time: \(linearEnd - linearStart)")
    
//    for (t, x, y, z) in trajectories {
//        if X.isEmpty {
//            tSpace = t
//            X = x.map { $0 / Double(realizations) }
//            Y = y.map { $0 / Double(realizations) }
//            Z = z.map { $0 / Double(realizations) }
//        } else {
//            for i in 0..<X.count {
//                X[i] += x[i] / Double(realizations)
//                Y[i] += y[i] / Double(realizations)
//                Z[i] += z[i] / Double(realizations)
//            }
//        }
//    }
    
    return (tSpace, X, Y, Z)
}

private func _decoherenceFunction(t: Double, A: Double, omegaC: Double) -> Complex<Double> {
    Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmegaSquared(omega: omega, A: A, omegaC: omegaC) * Complex(1 - .cos(omega * t), -.sin(omega * t))
    }
}

private func _decoherenceFunction(t: Double, G: [Complex<Double>], W: [Complex<Double>]) -> Complex<Double> {
    var result: Complex<Double> = .zero
    for (g, w) in zip(G.conjugate, W.conjugate) {
        result += g / (w * w) * (.exp(-t * w) - .one)
    }
    return result
}

func _ibmExact(endTime: Double) -> (tSpace: [Double], X: [Double], Y: [Double], Z: [Double]) {
    let A = 0.027
    let omegaC = 1.447
    
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, omegaC: omegaC)
    }
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        //let (G, W) = tPFD.fit(x: tSpace, y: bcf, realTerms: 4, imaginaryTerms: 4)
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 7)
        return (G, W)
    }()
    
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    let initialRho = initialState.outer(initialState.conjugate)
    
    let tSpace: [Double] = .linearSpace(0, endTime, Swift.max(Int(endTime / 0.1), 100))
    var X: [Double] = []
    var Y: [Double] = []
    var Z: [Double] = []
    let time = ContinuousClock().measure {
        tSpace.parallelMap { t in
            var rho: Matrix<Complex<Double>> = .init(elements: [initialRho[0, 0], initialRho[0, 1], initialRho[1, 0], initialRho[1, 1]], rows: 2, columns: 2)
            //let decoherenceFunction = _decoherenceFunction(t: t, A: A, omegaC: omegaC)
            let decoherenceFunction = _decoherenceFunction(t: t, G: G, W: W)
            rho[0, 1] = .exp(-decoherenceFunction) * rho[0, 1]
            rho[1, 0] = rho[0, 1].conjugate
            let x = 2 * rho[0, 1].real
            let y = 2 * rho[0, 1].imaginary
            let z = rho[0, 0].real - rho[1, 1].real
            return (x, y, z)
        }.forEach { x, y, z in
            X.append(x)
            Y.append(y)
            Z.append(z)
        }
    }
    print("Exact time: \(time)")
    return (tSpace, X, Y, Z)
    
}

public func _linearNMQSDIBM(endTime: Double = 7.0, realizations: Int, plotBCF: Bool = false) -> (tSpace: [Double], X: [Double], Y: [Double], Z: [Double]) {
    let A = 0.027
    let omegaC = 1.447
    
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, omegaC: omegaC)
    }
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        //let (G, W) = tPFD.fit(x: tSpace, y: bcf, realTerms: 4, imaginaryTerms: 4)
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        if plotBCF {
            let tSpace = [Double].linearSpace(0, 20, 501)
            let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
            let bcfExponentials = tSpace.map { t in
                var result: Complex<Double> = .zero
                for i in 0..<G.count {
                    result += G[i] * .exp(-t * W[i])
                }
                return result
            }
            plt.figure()
            plt.plot(x: tSpace, y: bcf.real, label: "Real")
            plt.plot(x: tSpace, y: bcf.imaginary, label: "Imaginary")
            plt.plot(x: tSpace, y: bcfExponentials.real, label: "Real", linestyle: "--")
            plt.plot(x: tSpace, y: bcfExponentials.imaginary, label: "Imaginary", linestyle: "--")
            plt.legend()
            plt.show()
            plt.close()
        }
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let nmqsd = NMQSDCalculation(dimension: 2, L: L, G: G, W: W)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime, dtMax: 0.01) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    nonisolated(unsafe) var tSpace: [Double] = []
    nonisolated(unsafe) var X: [Double] = []
    nonisolated(unsafe) var Y: [Double] = []
    nonisolated(unsafe) var Z: [Double] = []
    
    let mutex: Mutex<Void> = Mutex(())
    let bcfIntegral: @Sendable (Double) -> Complex<Double> = { t in
        var result: Complex<Double> = .zero
        for (g, w) in zip(G, W) {
            result += g / w * (.one - .exp(-t * w))
        }
        return result
    }
//    let OBar: @Sendable (Double, GaussianFFTNoiseProcess) -> Matrix<Complex<Double>> = { t, _ in
//        bcfIntegral(t) * L
//    }
    
    let linearStart = ContinuousClock().now
    (0..<realizations).parallelForEach { index in
        defer {
            if (index + 1) % 100 == 0 {
                print("Trajectory \(index + 1)")
            }
        }
        var _L = L
        let noise = zGenerator.generate()
        let (_tSpace, trajectory) = nmqsd.solveLinear(end: endTime, initialState: initialState, H: H, z: noise, OBar: { t, _ in
            _L.copyElements(from: L)
            _L.multiply(by: bcfIntegral(t))
            return _L
        }, stepSize: 0.01)
        let rho = nmqsd.mapTrajectoryToDensityMatrix(trajectory)
        let x = rho.map { 2 * $0[0, 1].real }
        let y = rho.map { 2 * $0[0, 1].imaginary }
        let z = rho.map { $0[0, 0].real - $0[1, 1].real }
        
        mutex.withLock { _ in
            if tSpace.isEmpty {
                tSpace = _tSpace
                X = x.map { $0 / Double(realizations) }
                Y = y.map { $0 / Double(realizations) }
                Z = z.map { $0 / Double(realizations) }
            } else {
                for i in 0..<X.count {
                    X[i] += x[i] / Double(realizations)
                    Y[i] += y[i] / Double(realizations)
                    Z[i] += z[i] / Double(realizations)
                }
            }
        }
    }
    let linearEnd = ContinuousClock().now
    print("Linear NMQSD time: \(linearEnd - linearStart)")
    
    return (tSpace, X, Y, Z)
}
