//
//  NoiseGenerationExample.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 27.5.2025.
//

import HOPS
import NumericsExtensions
import PythonKitUtilities
import SebbuScience
import SebbuCollections

func plotGaussianNoiseVsBCF<T: NoiseProcessGenerator>(count: Int, generator: T, tSpace: [Double], bcf: (Double) -> Complex<Double>) where T.Process: ComplexNoiseProcess, T: Sendable {
    let s = tSpace[0]
    let generationStart = ContinuousClock.now
    let noises = generator.generateParallel(count: count)
    let generationEnd = ContinuousClock.now
    print("Noise generation took:", generationEnd - generationStart)
    let bcf = tSpace.map { bcf($0) }
    let meanBCF = tSpace.parallelMap { t in
        var result: Complex<Double> = .zero
        for (i, z) in noises.enumerated() {
            let zt = z(t)
            let zs = z(s)
            result += zt * zs.conjugate
        }
        return result / Double(count)
    }
    plt.figure()
    plt.plot(x: tSpace, y: bcf.real, label: "Re alpha(t - s)")
    plt.plot(x: tSpace, y: bcf.imaginary, label: "Im alpha(t - s)")
    
    plt.plot(x: tSpace, y: meanBCF.real, label: "Re <z(t)z(s)^*>")
    plt.plot(x: tSpace, y: meanBCF.imaginary, label: "Im <z(t)z(s)^*>")
    
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("BCF")
    plt.show()
    plt.close()
}

func plotGaussianMultiNoiseVsBCF<T: MultiNoiseProcessGenerator>(count: Int, generator: T, tSpace: [Double], bcfs: [(Double) -> Complex<Double>]) where T.Process: ComplexNoiseProcess, T: Sendable {
    let s = tSpace[0]
    let noises = generator.generateParallel(count: count)
    let bcfs = bcfs.map { bcf in tSpace.map { bcf($0) } }
    
    for i in 0..<bcfs.count {
        let noises = noises.map { $0[i] }
        let bcf = bcfs[i]
        let meanBCF = tSpace.parallelMap { t in
            var result: Complex<Double> = .zero
            for z in noises {
                let zt = z(t)
                let zs = z(s)
                result += zt * zs.conjugate
            }
            return result / Double(count)
        }
        plt.figure()
        plt.plot(x: tSpace, y: meanBCF.real, label: "Re <z(t)z(s)^*>")
        plt.plot(x: tSpace, y: meanBCF.imaginary, label: "Im <z(t)z(s)^*>")
        
        plt.plot(x: tSpace, y: bcf.real, label: "Re alpha(t - s)")
        plt.plot(x: tSpace, y: bcf.imaginary, label: "Im alpha(t - s)")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("BCF")
        plt.title("Multi noise \(i)")
        plt.show()
        plt.close()
    }
}

func noiseGenerationExample() {
    let A = 0.027
    let omegaC = 1.447
    let tSpace = [Double].linearSpace(0, 10, 1000)
    let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0) }
    let bcfSpline = CubicHermiteSpline(x: tSpace, y: bcf)
    let generator = GaussianFFTNoiseProcessGenerator(tMax: tSpace.last!) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    plotGaussianNoiseVsBCF(count: 10000, generator: generator, tSpace: tSpace) { t in
        bcfSpline.sample(t)
    }
}

func multiNoiseGenerationExample() {
    let A = [0.027, 0.1, 0.4]
    let omegaC = [1.447, 2.0, 1.0]
    let tSpace = [Double].linearSpace(0, 10, 1000)
    let bcfs = zip(A, omegaC).map { a, ksi in tSpace.map { bathCorrelationFunction(A: a, omegaC: ksi, t: $0) } }
    let bcfSplines = bcfs.map { CubicHermiteSpline(x: tSpace, y: $0) }
    let bcfClosures = bcfSplines.map { spline in
        { (_ t: Double) -> Complex<Double> in
            spline.sample(t)
        }
    }
    let generator = GaussianFFTMultiNoiseProcessGenerator(tMax: tSpace.last!) { omega in
            .diagonal(from: zip(A, omegaC).map { a, ksi in Complex(spectralDensity(omega: omega, A: a, omegaC: ksi)) })
    }
    plotGaussianMultiNoiseVsBCF(count: 5000, generator: generator, tSpace: tSpace, bcfs: bcfClosures)
}

func ornsteinUhlenbeckExample() {
    let G = 0.27
    let W = Complex(1, 5)
    let tSpace = [Double].linearSpace(0, 20, 1000)
    let bcf = tSpace.map { G * .exp(-$0 * W) }
    let bcfSpline = CubicHermiteSpline(x: tSpace, y: bcf)
    let sampleSpace = [Double].linearSpace(0, 20, 10000)
    let generator = PreSampledOrnsteinUhlenbeckProcessGenerator(G: G, W: W, t: sampleSpace)
    plotGaussianNoiseVsBCF(count: 10000, generator: generator, tSpace: tSpace) { t in
        bcfSpline.sample(t)
    }
    
    let generator2 = GaussianFFTNoiseProcessGenerator(tMax: 20) { omega in
        G / .pi * (1.0 / (W - Complex(imaginary: omega))).real
    }
    plotGaussianNoiseVsBCF(count: 10000, generator: generator2, tSpace: tSpace) { t in
        bcfSpline.sample(t)
    }
}

func ornsteinUhlenbeckExample2() {
    let G = [0.27, 0.12]
    let W = [Complex(1, 5), Complex(2, -3)]
    let tSpace: [Double] = .linearSpace(0, 20, 1000)
    let bcf = tSpace.map { t in
        var result: Complex<Double> = .zero
        for (g, w) in zip(G, W) {
            result += g * .exp(-w * t)
        }
        return result
    }
    let bcfSpline = CubicHermiteSpline(x: tSpace, y: bcf)
    let sampleSpace = [Double].linearSpace(0, 20, 10000)
    let generator = PreSampledOrnsteinUhlenbeckProcessGenerator(G: G, W: W, t: sampleSpace)
    plotGaussianNoiseVsBCF(count: 10000, generator: generator, tSpace: tSpace) { t in
        bcfSpline.sample(t)
    }
    
    let noise = generator.generate()
    let samples = sampleSpace.map { noise.sample($0) }
    plt.figure()
    plt.plot(x: sampleSpace, y: samples.real, label: "Re z_t")
    plt.plot(x: sampleSpace, y: samples.imaginary, label: "Im z_t")
    plt.show()
    plt.close()
    
    let generator2 = GaussianFFTNoiseProcessGenerator(tMax: 20) { omega in
        var J: Double = .zero
        for (g, w) in zip(G, W) {
            var result = g / .pi
            result *= 1.0 / (w - Complex(imaginary: omega)).real
            J += result
        }
        //G / .pi * (1.0 / (W - Complex(imaginary: omega))).real
        return J
    }
    plotGaussianNoiseVsBCF(count: 10000, generator: generator2, tSpace: tSpace) { t in
        bcfSpline.sample(t)
    }
}

func ornsteinUhlenbeckExample3() {
    print("Correlated OU-process")
    func _bcfFunc(t: Double, spectralDensity: (Double) -> Double) -> Complex<Double> {
        Quad.integrate(a: 0, b: .infinity) { omega in
            spectralDensity(omega) * .init(length: 1, phase: -omega * t)
        }
    }
    let tMax: Double = 100.0
    let tSpace: [Double] = .linearSpace(0, 20, 1000)
    let bcf = tSpace.map { t in
        _bcfFunc(t: t) { omega in
            0.027 * omega * omega * omega * .exp(-omega * omega / (1.447 * 1.447))
        }
    }
    let bcfSpline = CubicHermiteSpline(x: tSpace, y: bcf)
    let sampleSpace = [Double].linearSpace(0, tMax, 0.01)
    let (G, W, r) = NonLinearFit.fitPhysical(t: tSpace, y: bcf, terms: 3)
    let generator = PreSampledCorrelatedOrnsteinUhlenbeckProcessGenerator(r: r, W: W, start: 0, end: tMax, dt: 0.01)
    plotGaussianNoiseVsBCF(count: 10000, generator: generator, tSpace: tSpace) { t in
        var result: Complex<Double> = .zero
        for (g, w) in zip(G, W) {
            result += g * .exp(-w * t)
        }
        return result
    }
    
    for _ in 0..<10 {
        let z = generator.generate()
        let noiseSamples = sampleSpace.map { z.sample($0) }
        plt.figure()
        plt.plot(x: sampleSpace, y: noiseSamples.real, label: "Re z")
        plt.plot(x: sampleSpace, y: noiseSamples.imaginary, label: "Im z")
        plt.show()
        plt.close()
    }
    
    let generator2 = GaussianFFTNoiseProcessGenerator(tMax: tMax, dtMax: 0.01) { omega in
        0.027 * omega * omega * omega * .exp(-omega * omega / (1.447 * 1.447))
    }
    plotGaussianNoiseVsBCF(count: 10000, generator: generator2, tSpace: tSpace) { t in
        bcfSpline.sample(t)
    }
    for _ in 0..<10 {
        let z = generator2.generate()
        let noiseSamples = sampleSpace.map { z.sample($0) }
        plt.figure()
        plt.plot(x: sampleSpace, y: noiseSamples.real, label: "Re z")
        plt.plot(x: sampleSpace, y: noiseSamples.imaginary, label: "Im z")
        plt.show()
        plt.close()
    }
}
