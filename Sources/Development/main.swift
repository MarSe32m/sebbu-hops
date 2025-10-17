//
//  main.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import PythonKit
import PythonKitUtilities

#if os(macOS)
PythonLibrary.useLibrary(at: "/Library/Frameworks/Python.framework/Versions/3.12/Python")
#elseif os(Linux)
//PythonLibrary.useLibrary(at: "/usr/lib64/libpython3.11.so.1.0")
PythonLibrary.useLibrary(at: "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0")
#elseif os(Windows)
//TODO: Set the library path on Windows machine
#endif

//radiativeDampingPlusPumpingExample(realizations: 400_000, endTime: 750.0)
//radiativeDampingExample(realizations: 1<<17, endTime: 750.0)
//testHOPSDecomposedPropagation(endTime: 120.0, at: 3.5)
//QFunctionPlot(endTime: 10.0)
//HOPSvsNMQSD(realizations: 1, endTime: 1750.0)
//IBMExample(realizations: 8196 * 2, endTime: 10.0, plotBCF: true)
//noiseGenerationExample()
//multiNoiseGenerationExample()

import SebbuScience
import Algorithms

func basisVectors(dimension: Int) -> [Vector<Complex<Double>>] {
    var result: [Vector<Complex<Double>>] = []
    for i in 0..<dimension {
        var basisVector: Vector<Complex<Double>> = .zero(dimension)
        basisVector[i] = .one
        result.append(basisVector)
    }
    return result
}


func partialTrace<T: AlgebraicField>(_ A: Matrix<T>, dimensions: [Int], keep: [Int]) -> Matrix<T> {
    let totalDim = dimensions.reduce(1, *)
    precondition(A.rows == totalDim)
    precondition(A.columns == totalDim)
    if keep.isEmpty { return .init(elements: [A.trace], rows: 1, columns: 1) }
    let n = dimensions.count
    let traced = (0..<n).filter { !keep.contains($0) }
    
    let keptDims = keep.map { dimensions[$0] }
    let reducedDim = keptDims.reduce(1, *)
    var rhoReduced: Matrix<T> = .zeros(rows: reducedDim, columns: reducedDim)
    //var rhoReduced = [T](repeating: .zero, count: reducedDim * reducedDim)
    
    let tracedDims = traced.map { dimensions[$0] }
    let tracedTotal = tracedDims.reduce(1, *)
    
    var rowFull = [Int](repeating: 0, count: n)
    var colFull = [Int](repeating: 0, count: n)
    
    @inline(__always)
    func unravelIndex(_ index: Int, dims: [Int]) -> [Int] {
        var result = [Int](repeating: 0, count: dims.count)
        var idx = index
        for i in (0..<dims.count).reversed() {
            result[i] = idx % dims[i]
            idx /= dims[i]
        }
        return result
    }
    
    @inline(__always)
    func ravelIndex(_ multi: [Int], dims: [Int]) -> Int {
        var flat = 0
        for i in 0..<dims.count {
            flat = flat * dims[i] + multi[i]
        }
        return flat
    }
    
    for aFlat in 0..<reducedDim {
        let aMultiKept = unravelIndex(aFlat, dims: keptDims)
        for bFlat in 0..<reducedDim {
            let bMultiKept = unravelIndex(bFlat, dims: keptDims)
            
            var sum: T = .zero
            
            // Sum over traced indices
            for t in 0..<tracedTotal {
                let tMulti = unravelIndex(t, dims: tracedDims)
                
                var ki = 0
                var ti = 0
                for i in 0..<n {
                    if keep.contains(i) {
                        rowFull[i] = aMultiKept[ki]
                        colFull[i] = bMultiKept[ki]
                        ki += 1
                    } else {
                        rowFull[i] = tMulti[ti]
                        colFull[i] = tMulti[ti]
                        ti += 1
                    }
                }
                
                let rowFlat = ravelIndex(rowFull, dims: dimensions)
                let colFlat = ravelIndex(colFull, dims: dimensions)
                sum += A[rowFlat, colFlat]
            }
            rhoReduced[aFlat, bFlat] = sum
            //rhoReduced[aFlat * reducedDim + bFlat] = sum
        }
    }
    
    //return .init(elements: rhoReduced, rows: reducedDim, columns: reducedDim)
    return rhoReduced
}

let rho1: Matrix<Complex<Double>> = .init(elements: [
    Complex(0.2), Complex(-0.1, 0.1),
    Complex(-0.1, -0.1), Complex(0.8)
], rows: 2, columns: 2)

let rho2: Matrix<Complex<Double>> = .init(elements: [
    Complex(0.5), Complex(0.5),
    Complex(0.5), Complex(0.5)
], rows: 2, columns: 2)
let rho3: Matrix<Complex<Double>> = .init(elements: [
    .zero, .zero,
    .zero, .one
], rows: 2, columns: 2)
let totalRho = rho1.kronecker(rho2.kronecker(rho3))

let _rho1 = partialTrace(totalRho, dimensions: [2,2,2], keep: [0])
let _rho2 = partialTrace(totalRho, dimensions: [2,2,2], keep: [1])
let _rho3 = partialTrace(totalRho, dimensions: [2,2,2], keep: [2])

print(_rho1,  rho1)
print(_rho2,  rho2)
print(_rho3,  rho3)
