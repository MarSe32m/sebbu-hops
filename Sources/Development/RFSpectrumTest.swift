//
//  RFSpectrumTest.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 26.10.2025.
//

import SebbuScience
import HOPS
import PythonKit
import PythonKitUtilities

private func spectralDensity(omega: Double, a: Double, ksi: Double) -> Double {
    a * omega * omega * omega * .exp(-(omega * omega) / (ksi * ksi))
}

private func spectralDensityByOmega(omega: Double, a: Double, ksi: Double) -> Double {
    a * omega * omega * .exp(-(omega * omega) / (ksi * ksi))
}

private func batchCorrelationFunction(tau: Double, temperature: Double, spectralDensity: (_ omega: Double) -> Double) -> Complex<Double> {
    let beta = temperature == .zero ? .infinity : (1 / temperature)
    return Quad.integrate(a: 0, b: .infinity) { omega in
        let J = spectralDensity(omega)
        return Complex(
            J * Double.coth(beta * omega / 2.0) * Double.cos(-omega * tau),
            J                                   * Double.sin(-omega * tau)
        )
    }
}

func testRFSpectrum(omegaX: Double, detuning: Double, omegaC: Double, rabi: Double,
                    a: Double, ksi: Double, kappa: Double, gammaR: Double, temperature: Double,
                    steadyStateTime: Double, endTime: Double, trajectories: Int) {
    // Parameters
    let renormalizationEnergy = Quad.integrate(a: .zero, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, a: a, ksi: ksi)
    }
    let omegaDrive = omegaX - detuning - renormalizationEnergy
    let OmegaTilde = rabi / (omegaC * 2)
    let delta = omegaX - omegaDrive
    
    // System operators
    let sigmaX: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .one, .zero], rows: 2, columns: 2)
    let sigmaY: Matrix<Complex<Double>> = .init(elements: [.zero, -.i, .i, .zero], rows: 2, columns: 2)
    let sigmaMinus: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    let sigmaPlus: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .one, .zero], rows: 2, columns: 2)
    let sigmaPlusSigmaMinus = sigmaPlus.dot(sigmaMinus)
    
    // Hamiltonian
    let H = delta * sigmaPlusSigmaMinus + OmegaTilde * omegaC * sigmaX + OmegaTilde * kappa / 2 * sigmaY - 0.5 * gammaR * .i * sigmaPlusSigmaMinus
    
    // Environment coupling operator
    let L = sigmaPlusSigmaMinus
    
    // Noise generators
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
        spectralDensity(omega: omega, a: a, ksi: ksi)
    }
    let whiteNoiseGenerator = PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gammaR / 2, start: 0, end: endTime, step: 0.01)
    
    // Bath correlation function exponential expansion
    let (G, W) = {
        let tSpace = [Double].linearSpace(0, 20, 501)
        let bcf = tSpace.map { tau in
            batchCorrelationFunction(tau: tau, temperature: temperature) { omega in
                spectralDensity(omega: omega, a: a, ksi: ksi)
            }
        }
        return MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
    }()
    
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 6)
    
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    var tauSpace: [Double] = []
    var correlationFunction: [Complex<Double>] = []
    
    var tSpace: [Double] = []
    var rho: [Matrix<Complex<Double>>] = []
    
    var sigmaPlusExpSS: Complex<Double> = .zero
    
    let batchSize = 512
    var trajectoriesComputed = 0
    
    while trajectoriesComputed < trajectories {
        let trajectoriesToCompute = Swift.max(batchSize, Swift.min(trajectories - trajectoriesComputed, batchSize))
        defer {
            trajectoriesComputed += trajectoriesToCompute
            print(trajectoriesComputed, "/", trajectories)
        }
        
        let z = zGenerator.generateParallel(count: trajectoriesToCompute / 2)
        let w = whiteNoiseGenerator.generateParallel(count: trajectoriesToCompute / 2)
        
        let trajs = zip(z, w).parallelMap { z, w in
            var _O: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
            let customOperator: (Double, Vector<Complex<Double>>, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, bra, ket in
                let sigmaPlusExp = ket.inner(ket, metric: sigmaPlus) / ket.normSquared
                _O.zeroElements()
                _O.add(sigmaPlus, multiplied: gammaR * sigmaPlusExp)
                return _O
            }
            let (tauSpace1, braTrajectory1, ketTrajectory1, _, normalizationFactor1) = hierarchy.solveNonLinearTwoTimeCorrelationFunction(t: steadyStateTime, A: sigmaPlus, s: endTime, B: sigmaMinus, initialState: initialState, H: H, z: z, whiteNoise: w, diffusionOperator: sigmaMinus, braCustomOperators: [customOperator], ketCustomOperators: [customOperator], stepSize: 0.1)
            let (tauSpace2, braTrajectory2, ketTrajectory2, _, normalizationFactor2) = hierarchy.solveNonLinearTwoTimeCorrelationFunction(t: steadyStateTime, A: sigmaPlus, s: endTime, B: sigmaMinus, initialState: initialState, H: H, z: z.antithetic(), whiteNoise: w.antithetic(), diffusionOperator: sigmaMinus, braCustomOperators: [customOperator], ketCustomOperators: [customOperator], stepSize: 0.1)
            
            var _tSpace1: [Double] = []
            var _rho1: [Matrix<Complex<Double>>] = []
            var _perturbedTSpace1: [Double] = []
            var _rhoPerturbed1: [Matrix<Complex<Double>>] = []
            var _tauSpace1: [Double] = []
            var _correlationFunction1: [Complex<Double>] = []
            for (i, tau) in tauSpace1.enumerated() {
                _tSpace1.append(tau)
                _rho1.append(ketTrajectory1[i].outer(braTrajectory1[i].conjugate) / normalizationFactor1[i])
                if tau < steadyStateTime {
                    continue
                } else {
                    _tauSpace1.append(tau - steadyStateTime)
                    _correlationFunction1.append(braTrajectory1[i].inner(ketTrajectory1[i], metric: sigmaMinus) / normalizationFactor1[i])
                }
            }
            let rhoSpline1 = CubicHermiteSpline(x: _tSpace1, y: _rho1)
            let spline1 = LinearInterpolator(x: _tauSpace1, y: _correlationFunction1)
            
            var _tSpace2: [Double] = []
            var _rho2: [Matrix<Complex<Double>>] = []
            
            var _tauSpace2: [Double] = []
            var _correlationFunction2: [Complex<Double>] = []
            for (i, tau) in tauSpace2.enumerated() {
                _tSpace2.append(tau)
                _rho2.append(ketTrajectory2[i].outer(braTrajectory2[i].conjugate) / normalizationFactor2[i])
                if tau < steadyStateTime {
                    continue
                } else {
                    _tauSpace2.append(tau - steadyStateTime)
                    _correlationFunction2.append(braTrajectory2[i].inner(ketTrajectory2[i], metric: sigmaMinus) / normalizationFactor2[i])
                }
            }
            let rhoSpline2 = CubicHermiteSpline(x: _tSpace2, y: _rho2)
            let spline2 = LinearInterpolator(x: _tauSpace2, y: _correlationFunction2)
            
            let _tSpace: [Double] = _tSpace1
            let rhoSamples = _tSpace.map { (rhoSpline1.sample($0) + rhoSpline2.sample($0))}
            let rhoSpline = CubicHermiteSpline(x: _tSpace, y: rhoSamples)
            
            let _tauSpace: [Double] = _tauSpace1
            let samples = _tauSpace.map { (spline1($0) + spline2($0))}
            let correlationFunctionSpline = LinearInterpolator(x: _tauSpace, y: samples)
            return (correlationFunctionSpline, rhoSpline)
        }
        for (correlationFunctionInterpolator, rhoInterpolator) in trajs {
            sigmaPlusExpSS += rhoInterpolator.sample(steadyStateTime - 0.1).dot(sigmaPlus).trace / Double(trajectories)
            if tSpace.isEmpty {
                tSpace = rhoInterpolator.x
                rho = rhoInterpolator.y.map { $0 / (Double(trajectories)) }
            } else {
                for i in tSpace.indices {
                    rho[i] += rhoInterpolator.sample(tSpace[i]) / (Double(trajectories))
                }
            }
            
            if tauSpace.isEmpty {
                tauSpace = correlationFunctionInterpolator.x
                correlationFunction = correlationFunctionInterpolator.y.map { $0 / (Double(trajectories)) }
            } else {
                for i in tauSpace.indices {
                    correlationFunction[i] += correlationFunctionInterpolator(tauSpace[i]) / (Double(trajectories))
                }
            }
        }
    }
    plt.figure()
    plt.plot(x: tSpace, y: rho.map { $0[0, 0] - $0[1, 1] }.real, label: "<z>")
    plt.plot(x: tSpace, y: rho.map { 2.0 * $0[0, 1] }.real, label: "<x>")
    plt.plot(x: tSpace, y: rho.map { 2.0 * $0[0, 1] }.imaginary, label: "<y>")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("<O>")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(x: tauSpace, y: correlationFunction.map { $0 - sigmaPlusExpSS.lengthSquared }.real, label: "Re <+ -(t)>")
    plt.plot(x: tauSpace, y: correlationFunction.map { $0 - sigmaPlusExpSS.lengthSquared }.imaginary, label: "Im <+ -(t)>")
    plt.plot(x: [0, tauSpace.last!], y: [sigmaPlusExpSS.lengthSquared, sigmaPlusExpSS.lengthSquared])
    plt.legend()
    plt.show()
    plt.close()
    
    let __tauSpace = tauSpace
    let __correlationFunction = correlationFunction.map { $0 - sigmaPlusExpSS.lengthSquared }
    let omegaSpace = [Double].linearSpace(-3 * rabi, 3 * rabi, 6000)
    let S = omegaSpace.parallelMap { omega in
        2.0 * Trapezoid.integrate(y: zip(__tauSpace, __correlationFunction).map { $0.1 * Complex(length: 1, phase: omega * $0.0) }, x: __tauSpace)
    }.real
    let max = S.max()!
    plt.figure()
    plt.plot(x: omegaSpace.map { $0 / rabi }, y: S.map { $0 / max })
    plt.legend()
    plt.show()
    plt.close()
}
