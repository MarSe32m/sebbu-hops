//
//  UnifiedLinearHOPSStateFunc.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 28.5.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import BasicContainers

extension UnifiedHOPSHierarchy {
    @usableFromInline
    internal struct LinearHOPSStateFunc<Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess>: ~Copyable, ~Escapable, ODERHSFunction, SDERHSFunction, HOPSStateFuncProtocol {
        
        public typealias NoiseType = Complex<Double>
        
        @usableFromInline
        internal let hierarchyPointer: UnsafePointer<UnifiedHOPSHierarchy>
        
        @usableFromInline
        var Heff: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        let H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void
        
        @usableFromInline
        let noises: Span<Noise>
        
        @usableFromInline
        let customOperators: Span<CustomOperator>
        
        @usableFromInline
        let jumpOperators: Span<JumpOperator<WhiteNoise>>
        
        @usableFromInline
        var LDaggerExpectationValues: UniqueVector<Complex<Double>>
        
        @usableFromInline
        let WConjugateVector: UniqueVector<Complex<Double>>
        
        @usableFromInline
        var noiseShifts: UniqueVector<Complex<Double>>
        
        @usableFromInline
        let dimension: Int
        
        @usableFromInline
        let totalDimension: Int
        
        @usableFromInline
        let shiftType: ShiftType
        
        @_lifetime(copy noises, copy customOperators, copy jumpOperators)
        @inlinable
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
            shiftType: ShiftType,
            customOperators: consuming Span<CustomOperator>,
            jumpOperators: consuming Span<JumpOperator<WhiteNoise>>
        ) {
            self.dimension = hierarchy.pointee.dimension
            self.totalDimension = hierarchy.pointee.totalDimension
            self.hierarchyPointer = hierarchy
            self.Heff = .zeros(rows: hierarchyPointer.pointee.dimension, columns: hierarchyPointer.pointee.dimension)
            self.H = H
            self.noises = noises
            self.customOperators = customOperators
            self.jumpOperators = jumpOperators
            self.LDaggerExpectationValues = .zero(hierarchy.pointee.LDagger.count)
            self.WConjugateVector = .init(count: hierarchy.pointee.W.count) { buffer in
                for i in hierarchy.pointee.W.indices {
                    buffer[i] = -hierarchy.pointee.W[i].conjugate
                }
            }
            self.noiseShifts = .zero(noises.count)
            self.shiftType = shiftType
        }
        
        @_lifetime(copy noises, copy customOperators)
        @inlinable
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
            shiftType: ShiftType,
            customOperators: consuming Span<CustomOperator>
        ) where WhiteNoise == ZeroNoiseProcess {
            self.dimension = hierarchy.pointee.dimension
            self.totalDimension = hierarchy.pointee.totalDimension
            self.hierarchyPointer = hierarchy
            self.Heff = .zeros(rows: hierarchyPointer.pointee.dimension, columns: hierarchyPointer.pointee.dimension)
            self.H = H
            self.noises = noises
            self.customOperators = customOperators
            self.jumpOperators = Span()
            self.LDaggerExpectationValues = .zero(hierarchy.pointee.LDagger.count)
            self.WConjugateVector = .init(count: hierarchy.pointee.W.count) { buffer in
                for i in hierarchy.pointee.W.indices {
                    buffer[i] = -hierarchy.pointee.W[i].conjugate
                }
            }
            self.noiseShifts = .zero(noises.count)
            self.shiftType = shiftType
        }
        
        @inlinable
        public mutating func evaluate(t: Double, y state: borrowing HOPSState, dy result: inout HOPSState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            let LDaggerSpan = hierarchyPointer.pointee.LDagger.span
            
            let systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: state.totalStateVector.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: self.Heff.elements, rows: dimension, columns: dimension)
            var LDaggerExpectations: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.LDaggerExpectationValues.components, count: LDaggerSpan.count)
            var noiseShifts: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.noiseShifts.components, count: noises.count)
            
            if shiftType == .meanField {
                // Noise / hierarchy shifting
                let normSquared = systemState.normSquared
                for i in 0..<LDaggerExpectations.count {
                    LDaggerExpectations[unchecked: i] = systemState.inner(metric: LDaggerSpan[i], systemState) / normSquared
                }
                hierarchyPointer.pointee.M.dot(LDaggerExpectations.components, into: result.shiftVector.components)
                for i in 0..<result.shiftVector.count {
                    result.shiftVector[unchecked: i] = Relaxed.multiplyAdd(WConjugateVector[unchecked: i], state.shiftVector[unchecked: i], result.shiftVector[unchecked: i])
                }
                for i in hierarchyPointer.pointee.shiftIndices.indices {
                    noiseShifts[unchecked: i] = .zero
                    for j in hierarchyPointer.pointee.shiftIndices[i] {
                        noiseShifts[unchecked: i] = Relaxed.sum(noiseShifts[unchecked: i], state.shiftVector[unchecked: j])
                    }
                }
            }
            
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            for i in hierarchyPointer.pointee.L.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i].sample(t).conjugate)
                if shiftType == .meanField {
                    Heff.add(LDaggerSpan[unchecked: i], multiplied: -noiseShifts[unchecked: i].conjugate)
                }
            }
            for i in 0..<customOperators.count {
                customOperators[unchecked: i](t, systemState, addingTo: &Heff)
            }
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            var kWIndex = 0
            while index < totalDimension {
                Heff.unsafeDot(currentStatePointer, into: resultPointer)
                let kW = kWSpan[unchecked: kWIndex]
                for i in 0..<dimension {
                    resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                }
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
                kWIndex &+= 1
            }
            hierarchyPointer.pointee.B.dot(state.totalStateVector.components, addingInto: result.totalStateVector.components)
            if shiftType == .meanField {
                for i in hierarchyPointer.pointee.N.indices {
                    hierarchyPointer.pointee.N[i].dot(state.totalStateVector.components, multiplied: -LDaggerExpectations[i].conjugate, addingInto: result.totalStateVector.components)
                }
            }
            let _ = systemState.consumeComponents()
            let _ = Heff.consumeElements()
            let _ = LDaggerExpectations.consumeComponents()
            let _ = noiseShifts.consumeComponents()
        }
        
        @inlinable
        func drift(t: Double, y state: borrowing HOPSState, into result: inout HOPSState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            let LDaggerSpan = hierarchyPointer.pointee.LDagger.span
            
            let systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: state.totalStateVector.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: Heff.elements, rows: dimension, columns: dimension)
            var LDaggerExpectations: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.LDaggerExpectationValues.components, count: LDaggerSpan.count)
            var noiseShifts: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.noiseShifts.components, count: noises.count)
            
            if shiftType == .meanField {
                // Noise / hierarchy shifting
                let normSquared = systemState.normSquared
                for i in 0..<LDaggerExpectations.count {
                    LDaggerExpectations[unchecked: i] = systemState.inner(metric: LDaggerSpan[i], systemState) / normSquared
                }
                
                hierarchyPointer.pointee.M.dot(LDaggerExpectations.components, into: result.shiftVector.components)
                for i in 0..<result.shiftVector.count {
                    result.shiftVector[unchecked: i] = Relaxed.multiplyAdd(WConjugateVector[unchecked: i], state.shiftVector[unchecked: i], result.shiftVector[unchecked: i])
                }
                for i in hierarchyPointer.pointee.shiftIndices.indices {
                    noiseShifts[unchecked: i] = .zero
                    for j in hierarchyPointer.pointee.shiftIndices[i] {
                        noiseShifts[unchecked: i] = Relaxed.sum(noiseShifts[unchecked: i], state.shiftVector[unchecked: j])
                    }
                }
            }
            
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            for i in hierarchyPointer.pointee.L.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i].sample(t).conjugate)
                if shiftType == .meanField {
                    Heff.add(LDaggerSpan[unchecked: i], multiplied: -noiseShifts[unchecked: i].conjugate)
                }
            }
            for i in 0..<customOperators.count {
                customOperators[unchecked: i](t, systemState, addingTo: &Heff)
            }
            for i in 0..<jumpOperators.count {
                Heff.add(jumpOperators[i].LDaggerL, multiplied: -0.5 * jumpOperators[i].rate(t))
            }
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            var kWIndex = 0
            while index < totalDimension {
                Heff.unsafeDot(currentStatePointer, into: resultPointer)
                let kW = kWSpan[unchecked: kWIndex]
                for i in 0..<dimension {
                    resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                }
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
                kWIndex &+= 1
            }
            hierarchyPointer.pointee.B.dot(state.totalStateVector.components, addingInto: result.totalStateVector.components)
            if shiftType == .meanField {
                for i in hierarchyPointer.pointee.N.indices {
                    hierarchyPointer.pointee.N[i].dot(state.totalStateVector.components, multiplied: -LDaggerExpectations[i].conjugate, addingInto: result.totalStateVector.components)
                }
            }
            let _ = systemState.consumeComponents()
            let _ = Heff.consumeElements()
            let _ = LDaggerExpectations.consumeComponents()
            let _ = noiseShifts.consumeComponents()
        }
        
        @inlinable
        mutating func diffusion(t: Double, y state: borrowing HOPSState, channel: Int, into result: inout HOPSState) {
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            while index < totalDimension {
                jumpOperators[unchecked: channel].operate(on: currentStatePointer, into: resultPointer)
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
            }
            result.shiftVector.zeroComponents()
        }
        
        @inlinable
        func sampleWhiteNoise(t: Double, noises: inout MutableSpan<NoiseType>) {
            for i in 0..<jumpOperators.count {
                noises[unchecked: i] = jumpOperators[i].noise.sample(t)
            }
        }
        
        @inlinable
        public func zero() -> HOPSState {
            return .init(dimension: totalDimension, shiftDimension: hierarchyPointer.pointee.G.count)
        }
    }
}
