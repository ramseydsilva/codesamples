'use strict';

var async = require('async'),
    breadcrumb = require('../helpers/breadcrumb'),
    post = require('../models/post'),
    city = require('../models/city'),
    category = require('../models/category'),
    restaurant = require('../models/restaurant'),
    moment = require('moment');

exports.posts = function(req, res) {
    var breadcrumbs = [ breadcrumb.home(), breadcrumb.posts('active') ];
    post.find().sort('-_id').populate('_city').populate('_restaurant').populate('_category').exec(function(err, posts){
        res.render('post/posts', {
            title: "Posts",
            breadcrumbs: breadcrumbs,
            posts: posts
        });
    });
};

exports.new = function(req, res) {
    var breadcrumbs = [ breadcrumb.home(), breadcrumb.posts(), breadcrumb.newPost() ];
    res.render('post/newPost', {
        title: 'New Post',
        breadcrumbs: breadcrumbs,
        city: req.query.city,
        category: req.query.category,
        restaurant: req.query.restaurant,
        item: req.query.item,
        album: req.query.album,
        albumId: req.query.albumId
    });
};

exports.post = function(req, res) {
    var post = res.locals.post;
    var breadcrumbs = [ breadcrumb.home(), breadcrumb.city(post._city), breadcrumb.restaurant(post._restaurant), breadcrumb.post(post, 'active') ];
    res.render('post/post', {
        title: "Post",
        breadcrumbs: breadcrumbs,
        post: post,
        userHasRights: post.userHasRights(req.user),
        time: moment(post.createdAt).fromNow()
    });
};

exports.edit = function(req, res) {
    var post = res.locals.post;
    var breadcrumbs = [ breadcrumb.home(), breadcrumb.city(post._city), breadcrumb.restaurant(post._restaurant), breadcrumb.post(post, 'active'),
        { text: 'Edit', url: post.editUrl, class: 'active'} ];

    res.render('post/edit', {
        title: "Edit Post",
        breadcrumbs: breadcrumbs,
        post: post,
        back: {
            text: 'Back to Post',
            href: post.url
        }
    });
};

exports.delete = function(req, res) {
    var post = res.locals.post;
    var breadcrumbs = [ breadcrumb.home(), breadcrumb.city(post._city), breadcrumb.restaurant(post._restaurant), breadcrumb.post(post, 'active'),
        { text: 'Delete', url: post.deleteUrl, class: 'active'} ];
    res.render('post/delete', {
        title: "Delete Post",
        breadcrumbs: breadcrumbs,
        post: post,
        back: {
            text: 'Back to Post',
            href: post.url
        }
    });
};