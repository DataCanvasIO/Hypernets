const config = {
    entry: ['./src/index.js'],
    output: {
        path: __dirname + '/build',
        filename: 'datUI.js',
        libraryTarget: "commonjs", // commonjs, var
        library: 'datUI'
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude:  /node_modules/,
                loader: 'babel-loader',
                options: {
                    presets: ['@babel/preset-env', '@babel/react'],
                    plugins: ['@babel/plugin-proposal-class-properties']
                }
            },
            {
                test: /\.css$/,
                use: [
                    'style-loader',
                    'css-loader'
                ]
            },
            {
                test: /\.(png|svg|jpg|gif)$/,
                use: [
                    'file-loader'
                ]
            }
        ]
    },
    resolve: {
        extensions: ['.js']
    },
    devServer:{
        writeToDisk:true,
        hot:false,
        inline: false,
    },
    mode: 'development'
};
module.exports = config;
