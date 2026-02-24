#include "qt_utils/ConsoleOutputWidget.hpp"

#include <QApplication>
#include <QClipboard>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

namespace qt_utils
{

ConsoleOutputWidget::ConsoleOutputWidget(QWidget* parent)
    : QWidget(parent)
{
    init(tr("Console Output"));
}

ConsoleOutputWidget::ConsoleOutputWidget(const QString& title, QWidget* parent)
    : QWidget(parent)
{
    init(title);
}

ConsoleOutputWidget::~ConsoleOutputWidget() = default;

void ConsoleOutputWidget::init(const QString& title)
{
    auto* mainLayout = new QVBoxLayout(this);

    // Header row: title label
    auto* headerLayout = new QHBoxLayout();
    titleLabel_ = new QLabel(title, this);
    titleLabel_->setStyleSheet(QStringLiteral("QLabel { font-weight: bold; }"));
    headerLayout->addWidget(titleLabel_);
    headerLayout->addStretch(1);
    mainLayout->addLayout(headerLayout);

    // Text display
    textEdit_ = new QPlainTextEdit(this);
    textEdit_->setReadOnly(true);
    textEdit_->setLineWrapMode(QPlainTextEdit::NoWrap);

    QFont consoleFont = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    textEdit_->setFont(consoleFont);
    textEdit_->setStyleSheet(
        QStringLiteral("QPlainTextEdit { background-color: #2b2b2b; color: #f0f0f0; }"));
    mainLayout->addWidget(textEdit_);

    // Button row: Clear, Copy
    auto* buttonLayout = new QHBoxLayout();
    clearButton_ = new QPushButton(tr("Clear"), this);
    copyButton_ = new QPushButton(tr("Copy"), this);

    connect(clearButton_, &QPushButton::clicked, this, &ConsoleOutputWidget::clear);
    connect(copyButton_, &QPushButton::clicked, this, &ConsoleOutputWidget::copyToClipboard);

    buttonLayout->addWidget(clearButton_);
    buttonLayout->addWidget(copyButton_);
    buttonLayout->addStretch(1);
    mainLayout->addLayout(buttonLayout);
}

void ConsoleOutputWidget::appendText(const QString& text)
{
    textEdit_->appendPlainText(text);

    // Auto-scroll to bottom
    auto* scrollBar = textEdit_->verticalScrollBar();
    scrollBar->setValue(scrollBar->maximum());
}

void ConsoleOutputWidget::clear()
{
    textEdit_->clear();
    emit cleared();
}

void ConsoleOutputWidget::copyToClipboard()
{
    QApplication::clipboard()->setText(textEdit_->toPlainText());
}

void ConsoleOutputWidget::setTitle(const QString& title)
{
    titleLabel_->setText(title);
}

auto ConsoleOutputWidget::text() const -> QString
{
    return textEdit_->toPlainText();
}

}  // namespace qt_utils
