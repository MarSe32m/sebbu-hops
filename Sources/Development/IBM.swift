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
func spectralDensityByOmegaSquared(omega: Double, A: Double, omegaC: Double) -> Double {
    A * omega * .exp(-omega * omega / (omegaC * omegaC))
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
    let A = 0.27
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
        let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory)
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
        let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory, normalize: true)
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

public func IBMExampleUnified(realizations: Int, endTime: Double = 7.0, plotBCF: Bool = false) {
    let A = 0.27
    let omegaC = 1.447
    
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, omegaC: omegaC)
    }
    
    let BCF = {
        let tSpace: [Double] = .linearSpace(0, 10, 501)
        return UnifiedHOPSHierarchy.BathCorrelationFunction(tSpace: tSpace, terms: 3) { t in
            bathCorrelationFunction(A: A, omegaC: omegaC, t: t)
        }
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let hierarchy = UnifiedHOPSHierarchy(dimension: 2, L: L, bathCorrelationFunctions: BCF, depth: 4)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    let noiseGenerationStart = ContinuousClock().now
    let noises = zGenerator.generateParallel(count: realizations)
    let noiseGenerationEnd = ContinuousClock().now
    print("Noise generation time: \(noiseGenerationEnd - noiseGenerationStart)")
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    //let _initialState: UniqueVector<Complex<Double>> = .init(copying: initialState)
    
    let linearStart = ContinuousClock().now
    let linearTrajectories = noises.parallelMap { z in
        hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, noises: z, stepSize: 0.01)
    }
    let linearEnd = ContinuousClock().now
    print("Linear time: \(linearEnd - linearStart)")
    let nonLinearStart = ContinuousClock().now
    let nonLinearTrajectories = noises.parallelMap { z in
        hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, noises: z, shiftType: .meanField, stepSize: 0.01)
    }
    let nonLinearEnd = ContinuousClock().now
    print("Non-linear time: \(nonLinearEnd - nonLinearStart)")
    
    let linearTSpace = linearTrajectories[0].tSpace
    var linearRho: [Matrix<Complex<Double>>] = []
    for trajectory in linearTrajectories {
        let _rho = trajectory.densityMatrix(normalized: false)
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
    for trajectory in nonLinearTrajectories {
        let _rho = trajectory.densityMatrix(normalized: true)
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

public func IBMFockStateAmplitudesExample(endTime: Double = 7.0) {
    let A = 0.027
    let omegaC = 1.447
    
    let bcfTerms = 3
    let hierarchyDepth = 4
    
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, omegaC: omegaC)
    }
    let BCF = {
        let tSpace: [Double] = .linearSpace(0, 10, 501)
        return UnifiedHOPSHierarchy.BathCorrelationFunction(tSpace: tSpace, terms: bcfTerms) { t in
            bathCorrelationFunction(A: A, omegaC: omegaC, t: t)
        }
    }()
    print(BCF)
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let hierarchy = UnifiedHOPSHierarchy(dimension: 2, L: L, bathCorrelationFunctions: BCF, depth: hierarchyDepth)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    let noise = GaussianFFTNoiseProcess(tMax: endTime, seed: 1239473214) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    let zeroNoise = ZeroNoiseProcess()
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    //let _initialState: UniqueVector<Complex<Double>> = .init(copying: initialState)
    
    let linearTrajectory = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, noises: noise, stepSize: 0.01, includeHierarchy: true)
    let linearZeroNoiseTrajectory = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, noises: zeroNoise, stepSize: 0.01, includeHierarchy: true)
    let nonLinearTrajectory = hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, noises: noise, stepSize: 0.01, includeHierarchy: true)
    let nonLinearZeroNoiseTrajectory = hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, noises: zeroNoise, stepSize: 0.01, includeHierarchy: true)
    let nonLinearTrajectoryShifted = hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, noises: noise, shiftType: .meanField, stepSize: 0.01, includeHierarchy: true)
    let nonLinearZeroNoiseTrajectoryShifted = hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, noises: zeroNoise, shiftType: .meanField, stepSize: 0.01, includeHierarchy: true)
    
    // Plot linear hiearchy occupations
    plt.figure()
    for fockStateNumber in 0...hierarchyDepth {
        for mode in 0..<bcfTerms {
            guard let fockStateAmplitudes = hierarchy.fockStateAmplitudes(for: linearTrajectory, mode: mode, fockState: fockStateNumber) else {
                fatalError("Couldn't obtain fock state amplitudes")
            }
            guard let fockStateAmplitudesZeroNoise = hierarchy.fockStateAmplitudes(for: linearZeroNoiseTrajectory, mode: mode, fockState: fockStateNumber) else { fatalError("Couldn't obtain fock state amplitudes") }
            plt.plot(x: linearTrajectory.tSpace, y: fockStateAmplitudes, label: "\(fockStateNumber)_\(mode + 1)")
            _plt.plot(linearZeroNoiseTrajectory.tSpace, fockStateAmplitudesZeroNoise, color: "gray", alpha: 0.5)
        }
    }
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Amplitudes")
    plt.title("Linear occupations")
    plt.show()
    plt.close()
    
    plt.figure()
    for mode in 0..<bcfTerms {
        var meanOccupation: [Double] = .init(repeating: .zero, count: linearTrajectory.tSpace.count)
        var meanOccupationZeroNoise: [Double] = .init(repeating: .zero, count: linearZeroNoiseTrajectory.tSpace.count)
        for fockStateNumber in 0...hierarchyDepth {
            guard let fockStateAmplitudes = hierarchy.fockStateAmplitudes(for: linearTrajectory, mode: mode, fockState: fockStateNumber) else {
                fatalError("Couldn't obtain fock state amplitudes")
            }
            guard let fockStateAmplitudeZeroNoise = hierarchy.fockStateAmplitudes(for: linearZeroNoiseTrajectory, mode: mode, fockState: fockStateNumber) else { fatalError("Couldn't obtain fock state amplitudes") }
            for i in meanOccupation.indices {
                meanOccupation[i] += Double(fockStateNumber) * fockStateAmplitudes[i]
            }
            for i in meanOccupationZeroNoise.indices {
                meanOccupationZeroNoise[i] += Double(fockStateNumber) * fockStateAmplitudeZeroNoise[i]
            }
        }
        plt.plot(x: linearTrajectory.tSpace, y: meanOccupation, label: "<n_\(mode + 1)>")
        _plt.plot(linearZeroNoiseTrajectory.tSpace, meanOccupationZeroNoise, color: "gray", alpha: 0.5)
    }
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<n_mu>")
    plt.title("Linear <n>")
    plt.show()
    plt.close()
    
    // Plot non-linear hiearchy occupations
    plt.figure()
    for fockStateNumber in 0...hierarchyDepth {
        for mode in 0..<bcfTerms {
            guard let fockStateAmplitudes = hierarchy.fockStateAmplitudes(for: nonLinearTrajectory, mode: mode, fockState: fockStateNumber) else {
                fatalError("Couldn't obtain fock state amplitudes")
            }
            guard let fockStateAmplitudesZeroNoise = hierarchy.fockStateAmplitudes(for: nonLinearZeroNoiseTrajectory, mode: mode, fockState: fockStateNumber) else { fatalError("Couldn't obtain fock state amplitudes") }
            plt.plot(x: nonLinearTrajectory.tSpace, y: fockStateAmplitudes, label: "\(fockStateNumber)_\(mode + 1)")
            _plt.plot(nonLinearZeroNoiseTrajectory.tSpace, fockStateAmplitudesZeroNoise, color: "gray", alpha: 0.5)
        }
    }
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Amplitudes")
    plt.title("Non-Linear occupations")
    plt.show()
    plt.close()
    
    plt.figure()
    for mode in 0..<bcfTerms {
        var meanOccupation: [Double] = .init(repeating: .zero, count: nonLinearTrajectory.tSpace.count)
        var meanOccupationZeroNoise: [Double] = .init(repeating: .zero, count: nonLinearTrajectory.tSpace.count)
        for fockStateNumber in 0...hierarchyDepth {
            guard let fockStateAmplitudes = hierarchy.fockStateAmplitudes(for: nonLinearTrajectory, mode: mode, fockState: fockStateNumber) else {
                fatalError("Couldn't obtain fock state amplitudes")
            }
            guard let fockStateAmplitudeZeroNoise = hierarchy.fockStateAmplitudes(for: nonLinearZeroNoiseTrajectory, mode: mode, fockState: fockStateNumber) else { fatalError("Couldn't obtain fock state amplitudes") }
            for i in meanOccupation.indices {
                meanOccupation[i] += Double(fockStateNumber) * fockStateAmplitudes[i]
            }
            for i in meanOccupationZeroNoise.indices {
                meanOccupationZeroNoise[i] += Double(fockStateNumber) * fockStateAmplitudeZeroNoise[i]
            }
        }
        plt.plot(x: nonLinearTrajectory.tSpace, y: meanOccupation, label: "<n_\(mode + 1)>")
        _plt.plot(nonLinearTrajectory.tSpace, meanOccupationZeroNoise, color: "gray", alpha: 0.5)
        
    }
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<n_mu>")
    plt.title("Non-Linear <n>")
    plt.show()
    plt.close()
    
    // Plot non-linear shifted hiearchy occupations
    plt.figure()
    for fockStateNumber in 0...hierarchyDepth {
        for mode in 0..<bcfTerms {
            guard let fockStateAmplitudes = hierarchy.fockStateAmplitudes(for: nonLinearTrajectoryShifted, mode: mode, fockState: fockStateNumber) else {
                fatalError("Couldn't obtain fock state amplitudes")
            }
            guard let fockStateAmplitudesZeroNoise = hierarchy.fockStateAmplitudes(for: nonLinearZeroNoiseTrajectoryShifted, mode: mode, fockState: fockStateNumber) else { fatalError("Couldn't obtain fock state amplitudes") }
            plt.plot(x: nonLinearTrajectoryShifted.tSpace, y: fockStateAmplitudes, label: "\(fockStateNumber)_\(mode + 1)")
            _plt.plot(nonLinearZeroNoiseTrajectoryShifted.tSpace, fockStateAmplitudesZeroNoise, color: "gray", alpha: 0.5)
        }
    }
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Amplitudes")
    plt.title("Non-Linear shifted occupations")
    plt.show()
    plt.close()
    
    plt.figure()
    for mode in 0..<bcfTerms {
        var meanOccupation: [Double] = .init(repeating: .zero, count: nonLinearTrajectoryShifted.tSpace.count)
        var meanOccupationZeroNoise: [Double] = .init(repeating: .zero, count: nonLinearTrajectoryShifted.tSpace.count)
        for fockStateNumber in 0...hierarchyDepth {
            guard let fockStateAmplitudes = hierarchy.fockStateAmplitudes(for: nonLinearTrajectoryShifted, mode: mode, fockState: fockStateNumber) else {
                fatalError("Couldn't obtain fock state amplitudes")
            }
            guard let fockStateAmplitudeZeroNoise = hierarchy.fockStateAmplitudes(for: nonLinearZeroNoiseTrajectoryShifted, mode: mode, fockState: fockStateNumber) else { fatalError("Couldn't obtain fock state amplitudes") }
            for i in meanOccupation.indices {
                meanOccupation[i] += Double(fockStateNumber) * fockStateAmplitudes[i]
            }
            for i in meanOccupationZeroNoise.indices {
                meanOccupationZeroNoise[i] += Double(fockStateNumber) * fockStateAmplitudeZeroNoise[i]
            }
        }
        plt.plot(x: nonLinearTrajectoryShifted.tSpace, y: meanOccupation, label: "<n_\(mode + 1)>")
        _plt.plot(nonLinearTrajectoryShifted.tSpace, meanOccupationZeroNoise, color: "gray", alpha: 0.5)
        
    }
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<n_mu>")
    plt.title("Non-Linear shifted <n>")
    plt.show()
    plt.close()
}
