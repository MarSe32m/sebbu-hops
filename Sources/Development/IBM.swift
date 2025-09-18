//
//  IBM.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import HOPS
import SebbuScience
import PythonKitUtilities

@inlinable
@inline(__always)
func spectralDensity(omega: Double, A: Double, omegaC: Double) -> Double {
    A * omega * omega * omega * .exp(-omega * omega / (omegaC * omegaC))
}

@inlinable
@inline(__always)
func spectralDensityByOmega(omega: Double, A: Double, omegaC: Double) -> Double {
    A * omega * omega * .exp(-omega * omega / (omegaC * omegaC))
}

@inlinable
@inline(__always)
func bathCorrelationFunction(A: Double, omegaC: Double, t: Double, temperature: Double = .zero) -> Complex<Double> {
    let beta: Double = temperature == .zero ? .infinity : 1 / temperature
    return Quad.integrate(a: 0, b: .infinity) { omega in
        Complex(
            Double.coth(beta * omega / 2.0) * spectralDensity(omega: omega, A: A, omegaC: omegaC) * Double.cos(-omega * t),
            spectralDensity(omega: omega, A: A, omegaC: omegaC) * Double.sin(-omega * t)
        )    
    }
}

public func IBMExample(realizations: Int, endTime: Double = 7.0, plotBCF: Bool = false) {
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
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 4)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    let noiseGenerationStart = ContinuousClock().now
    let noises = zGenerator.generateParallel(count: realizations)
    let noiseGenerationEnd = ContinuousClock().now
    print("Noise generation time: \(noiseGenerationEnd - noiseGenerationStart)")
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    let linearStart = ContinuousClock().now
    let linearTrajectories = noises.parallelMap { z in
        hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z, stepSize: 0.01)
    }
    let linearEnd = ContinuousClock().now
    print("Linear time: \(linearEnd - linearStart)")
    
    let nonLinearStart = ContinuousClock().now
    let nonLinearTrajectories = noises.parallelMap { z in
        hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z, stepSize: 0.01)
    }
    let nonLinearEnd = ContinuousClock().now
    print("Non-linear time: \(nonLinearEnd - nonLinearStart)")
    
    let linearTSpace = linearTrajectories[0].tSpace
    var linearRho: [Matrix<Complex<Double>>] = []
    for (_, trajectory) in linearTrajectories {
        let _rho = hierarchy.mapLinearToDensityMatrix(trajectory)
        if linearRho.isEmpty {
            linearRho = _rho.map { $0 / Double(realizations) }
        } else {
            for i in 0..<_rho.count {
                linearRho[i].add(_rho[i], multiplied: 1 / Double(realizations))
            }
        }
    }
    
    let nonLinearTSpace = nonLinearTrajectories[0].tSpace
    var nonLinearRho: [Matrix<Complex<Double>>] = []
    for (_, trajectory) in nonLinearTrajectories {
        let _rho = hierarchy.mapNonLinearToDensityMatrix(trajectory)
        if nonLinearRho.isEmpty {
            nonLinearRho = _rho.map { $0 / Double(realizations) }
        } else {
            for i in 0..<_rho.count {
                nonLinearRho[i].add(_rho[i], multiplied: 1 / Double(realizations))
            }
        }
    }
    
    let linearX = linearRho.map { 2 * $0[0, 1].real }
    let linearY = linearRho.map { 2 * $0[0, 1].imaginary }
    let linearZ = linearRho.map { $0[0, 0].real - $0[1, 1].real }
    
    let nonLinearX = nonLinearRho.map { 2 * $0[0, 1].real }
    let nonLinearY = nonLinearRho.map { 2 * $0[0, 1].imaginary }
    let nonLinearZ = nonLinearRho.map { $0[0, 0].real - $0[1, 1].real }
    
    
    plt.figure()
    plt.plot(x: linearTSpace, y: linearX, label: "Lin <x>")
    plt.plot(x: linearTSpace, y: linearY, label: "Lin <y>")
    plt.plot(x: linearTSpace, y: linearZ, label: "Lin <z>")
    
    plt.plot(x: nonLinearTSpace, y: nonLinearX, label: "Non-lin <x>", linestyle: "--")
    plt.plot(x: nonLinearTSpace, y: nonLinearY, label: "Non-lin <y>", linestyle: "--")
    plt.plot(x: nonLinearTSpace, y: nonLinearZ, label: "Non-lin <z>", linestyle: "--")
    
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("rho")
    plt.show()
    plt.close()
}
