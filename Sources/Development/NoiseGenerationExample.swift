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
    let noises = generator.generateParallel(count: count)
    let bcf = tSpace.map { bcf($0) }
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
    plt.show()
    plt.close()
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
