//
//  Swift Extensions.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 24.5.2026.
//

import Builtin

extension UnsafeMutablePointer {
    @inlinable
    @inline(always)
    func _unsafeCopy(from: Self, count: Int) {
        for i in 0..<count { self[i] = from[i] }
    }
}

#if swift(>=6.5)
#warning("TODO: Check whether theses Span / MutableSpan inititalizers are needed anymore")
#endif
extension Span where Element: ~Copyable {
    @inlinable
    @_lifetime(borrow value)
    public init(_ value: borrowing @_addressable Element) {
        let address = Builtin.unprotectedAddressOfBorrow(value)
        let span = unsafe Span(_unsafeStart: .init(address), count: 1)
        self = unsafe _overrideLifetime(span, borrowing: value)
    }
}

extension MutableSpan where Element: ~Copyable {
    @inlinable
    @_lifetime(&value)
    public init(_ value: inout Element) {
        let address = Builtin.unprotectedAddressOfBorrow(value)
        let span = unsafe MutableSpan(_unsafeStart: .init(address), count: 1)
        self = unsafe _overrideLifetime(span, mutating: &value)
    }
}
