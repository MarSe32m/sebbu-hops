// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "sebbu-hops",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "HOPS", targets: ["HOPS"]),
    ],
    dependencies: [
        .package(url: "https://github.com/MarSe32m/sebbu-science", branch: "main"),
        .package(url: "https://github.com/apple/swift-algorithms", branch: "main")
    ],
    targets: [
        .target(
            name: "HOPS",
            dependencies: [.product(name: "SebbuScience", package: "sebbu-science"),
                           .product(name: "Algorithms", package: "swift-algorithms")]
        ),
        .executableTarget(
            name: "Development",
            dependencies: [
                .target(name: "HOPS"),
                .product(name: "SebbuScience", package: "sebbu-science"),
                .product(name: "PythonKitUtilities", package: "sebbu-science")
            ]
        )
    ]
)
