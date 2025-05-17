//
//  LinearHOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import Algorithms

public extension HOPSHierarchy {
    
    @inlinable
    @inline(__always)
    func solveLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, customOperators: customOperators, stepSize: stepSize)
    }
    
    @inlinable
    @inline(__always)
    func solveLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<initialState.count {
            initialStateVector[i] = initialState[i]
        }
        var systemState = initialState
        var resultCache: Deque<Vector<Complex<Double>>> = .init(repeating: .zero(initialStateVector.count), count: 4)
        return withoutActuallyEscaping(H) { H in
            var Heff = H(start)
            var solver = RK45FixedStep(initialState: initialStateVector, t0: start, dt: stepSize) { t, currentState in
                for i in 0..<dimension {
                    systemState[i] = currentState[i]
                }
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                let z = z(t).conjugate
                Heff.add(L, multiplied: z)
                for customOperator in customOperators {
                    Heff.add(customOperator(t, systemState))
                }
                result.components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        var kWIndex = 0
                        while index < resultBuffer.count {
                            Heff.dot(currentStatePointer, into: resultPointer)
                            let kW = kWArray[kWIndex]
                            var i = 0
                            while i &+ 2 <= dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                                resultPointer[i &+ 1] = Relaxed.multiplyAdd(kW, currentStatePointer[i &+ 1], resultPointer[i &+ 1])
                                i &+= 2
                            }
                            while i < dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                                i &+= 1
                            }
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                            kWIndex &+= 1
                        }
                    }
                }
                B.dot(currentState, addingInto: &result)
                return result
            }
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(dimension), count: tSpace.capacity)
            var index = 0
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(dimension))
                }
                for i in 0..<dimension {
                    trajectory[index][i] = state[i]
                }
                tSpace.append(t)
                index += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }
    
    @inlinable
    @inline(__always)
    func solveLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, w: WhiteNoise, wOperator: Matrix<Complex<Double>>, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, w: w, wOperator: wOperator, customOperators: customOperators, stepSize: stepSize)
    }
    
    @inlinable
    @inline(__always)
    func solveLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, w: WhiteNoise, wOperator: Matrix<Complex<Double>>, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<initialState.count {
            initialStateVector[i] = initialState[i]
        }
        var systemState = initialState
        var resultCache: Deque<Vector<Complex<Double>>> = .init(repeating: .zero(initialStateVector.count), count: 4)
        return withoutActuallyEscaping(H) { H in
            var Heff = H(start)
            var solver = SRK2FixedStep(initialState: initialStateVector, t0: start, dt: stepSize) { t, currentState in
                for i in 0..<dimension {
                    systemState[i] = currentState[i]
                }
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                let z = z(t).conjugate
                Heff.add(L, multiplied: z)
                for customOperator in customOperators {
                    Heff.add(customOperator(t, systemState))
                }
                result.components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        var kWIndex = 0
                        while index < resultBuffer.count {
                            Heff.dot(currentStatePointer, into: resultPointer)
                            let kW = kWArray[kWIndex]
                            var i = 0
                            while i &+ 2 <= dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                                resultPointer[i &+ 1] = Relaxed.multiplyAdd(kW, currentStatePointer[i &+ 1], resultPointer[i &+ 1])
                                i &+= 2
                            }
                            while i < dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                                i &+= 1
                            }
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                            kWIndex &+= 1
                        }
                    }
                }
                B.dot(currentState, addingInto: &result)
                return result
            } g: { t, currentState in
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                result.components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        while index < resultBuffer.count {
                            wOperator.dot(currentStatePointer, into: resultPointer)
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                        }
                        
                    }
                }
                return result
            } w: { t in
                w(t)
            }
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(dimension), count: tSpace.capacity)
            var index = 0
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(dimension))
                }
                for i in 0..<dimension {
                    trajectory[index][i] = state[i]
                }
                tSpace.append(t)
                index += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }
}
